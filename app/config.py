import os


class Settings:
    vertex_project_id: str | None = os.getenv("VERTEX_PROJECT_ID")
    vertex_location: str = os.getenv("VERTEX_LOCATION", "us-central1")

    ng12_pdf_path: str = "data/ng12.pdf"
    vector_index_dir: str = "data/index"


settings = Settings()
