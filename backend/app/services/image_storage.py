from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings


class ImageStorageService:
    def __init__(self, upload_dir: Path = settings.UPLOAD_DIR) -> None:
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> dict[str, str | int]:
        extension = self._get_extension(file.filename)
        stored_filename = f"{uuid4().hex}{extension}"
        destination = self.upload_dir / stored_filename

        size_bytes = 0
        try:
            with destination.open("wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    size_bytes += len(chunk)
                    if size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                        destination.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="Uploaded image is too large.",
                        )
                    buffer.write(chunk)
        finally:
            await file.close()

        if size_bytes == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )

        return {
            "filename": stored_filename,
            "original_filename": file.filename or stored_filename,
            "content_type": file.content_type or "application/octet-stream",
            "size_bytes": size_bytes,
            "image_path": str(destination),
            "image_url": f"/uploads/{stored_filename}",
        }

    @staticmethod
    def _get_extension(filename: str | None) -> str:
        extension = Path(filename or "").suffix.lower()
        if extension in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return extension
        return ".jpg"
