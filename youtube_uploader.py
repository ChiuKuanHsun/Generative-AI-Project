"""
YouTube uploader — desktop OAuth flow for Shorts upload.

First run opens a browser for Google consent; the access token is cached in
youtube_token.json so subsequent uploads are silent.

Setup:
  1. client_secret.json must be in the project root (Desktop OAuth client).
  2. pip install google-auth-oauthlib google-api-python-client

Usage:
  from youtube_uploader import upload_video
  url = upload_video("output.mp4", "My title", "Description", privacy="public")
"""
from __future__ import annotations

import os

CLIENT_SECRET_PATH = "client_secret.json"
TOKEN_PATH         = "youtube_token.json"
SCOPES             = ["https://www.googleapis.com/auth/youtube.upload"]

CATEGORY_NEWS_AND_POLITICS = "25"


def _get_credentials():
    """Load cached creds, refresh if expired, otherwise run the consent flow."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as e:
            print(f"  Cached token unreadable ({e}); re-running consent flow.")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return creds
        except Exception as e:
            print(f"  Token refresh failed ({e}); re-running consent flow.")

    if not os.path.exists(CLIENT_SECRET_PATH):
        raise FileNotFoundError(
            f"OAuth client file not found: {CLIENT_SECRET_PATH}. "
            "Download it from Google Cloud Console (OAuth client → Desktop app)."
        )

    print("\n  Opening browser for Google consent...")
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    print(f"  Token saved to {TOKEN_PATH}")
    return creds


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    privacy: str = "public",
    category_id: str = CATEGORY_NEWS_AND_POLITICS,
) -> str:
    """Upload an mp4 to YouTube. Returns the Shorts URL."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if privacy not in ("public", "unlisted", "private"):
        raise ValueError(f"privacy must be public/unlisted/private, got {privacy!r}")

    creds   = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title":       title[:100],
            "description": description[:5000],
            "tags":        (tags or [])[:30],
            "categoryId":  category_id,
        },
        "status": {
            "privacyStatus":          privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        chunksize=-1,           # upload in a single request
        resumable=True,
        mimetype="video/mp4",
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    print(f"  Uploading {video_path} ({os.path.getsize(video_path) / 1e6:.1f} MB)...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"    {int(status.progress() * 100)}%")

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube did not return a video id: {response!r}")

    url = f"https://youtube.com/shorts/{video_id}"
    print(f"  Uploaded: {url}")
    return url


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python youtube_uploader.py <video_path> <title> [description]")
        sys.exit(1)
    video = sys.argv[1]
    title = sys.argv[2]
    desc  = sys.argv[3] if len(sys.argv) > 3 else "Test upload"
    print(upload_video(video, title, desc + " #Shorts", privacy="public"))
