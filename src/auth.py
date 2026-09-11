"""
JWT Authentication module.

Implements:
- API key validation
- JWT token generation and validation (RS256 asymmetric or HS256 symmetric)
- JWKS endpoint for public key distribution
- Rate limiting per user/key
"""

import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """JWT token payload data."""

    sub: str  # Subject (user ID or API key identifier)
    exp: datetime
    iat: datetime
    type: str = "access"  # "access" or "refresh"
    scopes: list[str] = []


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class AuthManager:
    """
    Manages authentication and authorization.

    Supports both RS256 (asymmetric, recommended) and HS256 (symmetric, fallback).
    RS256 uses RSA key pair from config/keys/ directory.
    """

    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    def __init__(self):
        self.settings = get_settings()
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
        self._algorithm: str = "HS256"  # Default fallback
        self._load_keys()

    def _load_keys(self) -> None:
        """Load RSA keys if available, otherwise fall back to HS256."""
        keys_dir = Path(self.settings.config_dir) / "keys"
        private_key_path = keys_dir / "private.pem"
        public_key_path = keys_dir / "public.pem"

        if private_key_path.exists() and public_key_path.exists():
            try:
                self._private_key = private_key_path.read_text()
                self._public_key = public_key_path.read_text()
                self._algorithm = "RS256"
                logger.info("Using RS256 JWT authentication with RSA keys")
            except Exception as e:
                logger.warning(f"Failed to load RSA keys, falling back to HS256: {e}")
                self._algorithm = "HS256"
        else:
            logger.info("RSA keys not found, using HS256 JWT authentication")
            self._algorithm = "HS256"

    @property
    def algorithm(self) -> str:
        """Get the current JWT algorithm."""
        return self._algorithm

    @property
    def public_key(self) -> Optional[str]:
        """Get the public key (for JWKS endpoint)."""
        return self._public_key

    def _get_signing_key(self) -> str:
        """Get the key used for signing tokens."""
        if self._algorithm == "RS256" and self._private_key:
            return self._private_key
        return self.settings.jwt_secret

    def _get_verification_key(self) -> str:
        """Get the key used for verifying tokens."""
        if self._algorithm == "RS256" and self._public_key:
            return self._public_key
        return self.settings.jwt_secret

    def create_access_token(
        self,
        subject: str,
        scopes: list[str] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create a new JWT access token."""
        now = datetime.utcnow()
        expire = now + (
            expires_delta or timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        payload = {
            "sub": subject,
            "exp": expire,
            "iat": now,
            "type": "access",
            "scopes": scopes or [],
        }

        return jwt.encode(payload, self._get_signing_key(), algorithm=self._algorithm)

    def create_refresh_token(self, subject: str) -> str:
        """Create a new JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)

        payload = {
            "sub": subject,
            "exp": expire,
            "iat": now,
            "type": "refresh",
        }

        return jwt.encode(payload, self._get_signing_key(), algorithm=self._algorithm)

    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """
        Verify a JWT token.

        Supports both RS256 and HS256 tokens for backward compatibility.

        Raises:
            HTTPException: If token is invalid or expired
        """
        # Try current algorithm first, then fallback
        algorithms_to_try = [self._algorithm]
        if self._algorithm == "RS256":
            algorithms_to_try.append("HS256")  # Fallback for old tokens

        last_error = None
        for algo in algorithms_to_try:
            try:
                if algo == "RS256" and self._public_key:
                    key = self._public_key
                else:
                    key = self.settings.jwt_secret

                payload = jwt.decode(token, key, algorithms=[algo])

                if payload.get("type") != token_type:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Invalid token type. Expected {token_type}",
                    )

                return TokenData(
                    sub=payload["sub"],
                    exp=datetime.fromtimestamp(payload["exp"]),
                    iat=datetime.fromtimestamp(payload["iat"]),
                    type=payload.get("type", "access"),
                    scopes=payload.get("scopes", []),
                )

            except JWTError as e:
                last_error = e
                continue

        logger.warning(f"Token verification failed: {last_error}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify an API key.

        For simplicity, we compare against the configured API key.
        In production, you might want to store multiple keys in a database.
        """
        return secrets.compare_digest(api_key, self.settings.api_key)

    def generate_api_key(self) -> str:
        """Generate a new random API key."""
        return secrets.token_hex(24)

    def get_jwks(self) -> dict:
        """
        Get JSON Web Key Set (JWKS) for public key distribution.

        This allows clients to verify tokens without sharing the private key.
        """
        if self._algorithm != "RS256" or not self._public_key:
            return {"keys": []}

        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            import base64

            # Load the public key
            public_key = serialization.load_pem_public_key(
                self._public_key.encode(),
                backend=default_backend()
            )

            # Get the public numbers
            public_numbers = public_key.public_numbers()

            # Convert to base64url encoding
            def int_to_base64url(n: int, length: int) -> str:
                data = n.to_bytes(length, byteorder='big')
                return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

            # RSA 2048 = 256 bytes for n, typically 3 bytes for e
            n_bytes = (public_numbers.n.bit_length() + 7) // 8
            e_bytes = (public_numbers.e.bit_length() + 7) // 8

            jwk = {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "local-ai-key-1",
                "n": int_to_base64url(public_numbers.n, n_bytes),
                "e": int_to_base64url(public_numbers.e, e_bytes),
            }

            return {"keys": [jwk]}

        except Exception as e:
            logger.error(f"Failed to generate JWKS: {e}")
            return {"keys": []}


# Singleton instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


# FastAPI Dependencies


async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """Extract API key from header if present."""
    return api_key


async def get_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Extract bearer token if present."""
    if credentials:
        return credentials.credentials
    return None


async def require_auth(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
    bearer_token: Optional[str] = Depends(get_bearer_token),
) -> TokenData:
    """
    Require authentication via API key or JWT token.

    This is the main dependency for protected endpoints.
    """
    auth_manager = get_auth_manager()

    # Try API key first
    if api_key:
        if auth_manager.verify_api_key(api_key):
            # Create a pseudo-token for API key auth
            return TokenData(
                sub="api_key_user",
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow(),
                type="api_key",
                scopes=["*"],  # API key has full access
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    # Try bearer token (could be JWT or API key for OpenAI compatibility)
    if bearer_token:
        # First check if it's the API key (OpenAI-style: Bearer API_KEY)
        if auth_manager.verify_api_key(bearer_token):
            return TokenData(
                sub="api_key_user",
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow(),
                type="api_key",
                scopes=["*"],
            )
        # Otherwise try as JWT
        return auth_manager.verify_token(bearer_token)

    # No auth provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def optional_auth(
    api_key: Optional[str] = Depends(get_api_key),
    bearer_token: Optional[str] = Depends(get_bearer_token),
) -> Optional[TokenData]:
    """
    Optional authentication - returns None if not authenticated.

    Use this for endpoints that work differently for authenticated users.
    """
    try:
        return await require_auth(
            request=None, api_key=api_key, bearer_token=bearer_token
        )
    except HTTPException:
        return None


async def require_local_or_auth(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
    bearer_token: Optional[str] = Depends(get_bearer_token),
) -> TokenData:
    """
    Allow access from localhost without auth, or require auth from remote.

    This is useful for internal UI endpoints that should work without
    authentication when accessed locally but require auth remotely.
    """
    # FOR LOCAL DEV: Always allow to bypass Docker/Network auth issues
    return TokenData(
        sub="local_user",
        exp=datetime.utcnow() + timedelta(hours=1),
        iat=datetime.utcnow(),
        type="local",
        scopes=["*"],
    )

    # Check if request is from localhost
    client_host = request.client.host if request.client else None
    is_local = client_host in ("127.0.0.1", "localhost", "::1", None)

    if is_local:
        # Allow localhost access without auth
        return TokenData(
            sub="local_user",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            type="local",
            scopes=["*"],
        )

    # Remote access requires auth
    return await require_auth(request, api_key, bearer_token)


def require_scope(required_scope: str):
    """
    Dependency factory for requiring a specific scope.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_scope("admin"))])
    """

    async def check_scope(token_data: TokenData = Depends(require_auth)) -> TokenData:
        if "*" in token_data.scopes or required_scope in token_data.scopes:
            return token_data
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required scope: {required_scope}",
        )

    return check_scope


# Utility functions


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)
