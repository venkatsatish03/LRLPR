from pydantic import BaseModel


class UploadImageResponse(BaseModel):
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    image_path: str
    image_url: str
