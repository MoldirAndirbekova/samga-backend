from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import bcrypt
import jwt

jwtSecret = os.environ.get("JWT_SECRET")

ACCESS_TOKEN_EXPIRES = timedelta(minutes=15) 
REFRESH_TOKEN_EXPIRES = timedelta(days=7)

def signJWT(user_id: str, is_refresh: bool = False) -> str:
    expires = datetime.now(tz=timezone.utc) + (REFRESH_TOKEN_EXPIRES if is_refresh else ACCESS_TOKEN_EXPIRES)

    payload = {
        "exp": expires,
        "userId": user_id,
        "type": "refresh" if is_refresh else "access", 
    }

    token = jwt.encode(payload, jwtSecret, algorithm="HS256")
    return token


def decodeJWT(token: str, is_refresh: bool = False) -> dict:
    try:
        decoded = jwt.decode(token, jwtSecret, algorithms=["HS256"])

        if decoded["type"] != ("refresh" if is_refresh else "access"):
            return None

        return decoded
    except jwt.ExpiredSignatureError:
        print("Token expired. Get a new one.")
        return None
    except jwt.InvalidTokenError:
        return None


def encryptPassword(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def validatePassword(password: str, encrypted: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), encrypted.encode("utf-8"))


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)

        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403, detail="Invalid or expired token.")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")

    def verify_jwt(self, jwtToken: str) -> bool:
        isTokenValid = False

        try:
            payload = decodeJWT(jwtToken)
        except:
            payload = None
        
        if payload:
            isTokenValid = True

        return isTokenValid
