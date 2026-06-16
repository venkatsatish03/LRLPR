from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from app.schemas.upload import UploadImageResponse
from app.services.image_storage import ImageStorageService

router = APIRouter(prefix="/images", tags=["images"])
storage_service = ImageStorageService()


@router.get("/upload", response_class=HTMLResponse)
def upload_form() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Upload License Plate Image</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 720px;
            margin: 48px auto;
            padding: 0 20px;
            line-height: 1.5;
          }
          form {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 24px;
          }
          button {
            margin-top: 16px;
            padding: 10px 16px;
            cursor: pointer;
          }
          code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
          }
        </style>
      </head>
      <body>
        <h1>Upload License Plate Image</h1>
        <p>This page sends a <code>POST</code> request to the upload endpoint.</p>
        <form action="/api/v1/images/upload" method="post" enctype="multipart/form-data">
          <label for="file">Choose image</label><br>
          <input id="file" name="file" type="file" accept="image/*" required><br>
          <button type="submit">Upload</button>
        </form>
        <p>API docs: <a href="/docs">/docs</a></p>
      </body>
    </html>
    """


@router.post(
    "/upload",
    response_model=UploadImageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...)) -> UploadImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image uploads are allowed.",
        )

    saved_image = await storage_service.save_upload(file)
    return UploadImageResponse(**saved_image)
