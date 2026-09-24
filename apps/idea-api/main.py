import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "todos.db"

app = FastAPI(title="我的待办事项")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ideas (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL,
                "desc"     TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )


init_db()


class TodoCreate(BaseModel):
    content: str = Field(min_length=1, max_length=200)


class TodoUpdate(BaseModel):
    content: Optional[str] = Field(default=None, min_length=1, max_length=200)
    done: Optional[bool] = None


class IdeaCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    desc: str = Field(min_length=1, max_length=500)


def row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "content": row["content"],
        "done": bool(row["done"]),
        "created_at": row["created_at"],
    }


@app.get("/api/todos")
def list_todos():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM todos ORDER BY done ASC, created_at DESC, id DESC"
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@app.post("/api/todos", status_code=201)
def create_todo(payload: TodoCreate):
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="内容不能为空")

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO todos (content, done, created_at) VALUES (?, 0, datetime('now', 'localtime'))",
            (content,),
        )
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return row_to_dict(row)


@app.patch("/api/todos/{todo_id}")
def update_todo(todo_id: int, payload: TodoUpdate):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="待办事项不存在")

        content = row["content"] if payload.content is None else payload.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="内容不能为空")
        done = row["done"] if payload.done is None else int(payload.done)

        conn.execute(
            "UPDATE todos SET content = ?, done = ? WHERE id = ?",
            (content, done, todo_id),
        )
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
    return row_to_dict(row)


@app.delete("/api/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int):
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="待办事项不存在")


def idea_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "desc": row["desc"],
        "created_at": row["created_at"],
    }


@app.get("/ideas")
def list_ideas():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM ideas ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [idea_row_to_dict(row) for row in rows]


@app.post("/ideas", status_code=201)
def create_idea(payload: IdeaCreate):
    title = payload.title.strip()
    desc = payload.desc.strip()
    if not title or not desc:
        raise HTTPException(status_code=400, detail="标题和描述不能为空")

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO ideas (title, "desc", created_at)
               VALUES (?, ?, datetime('now', 'localtime'))""",
            (title, desc),
        )
        row = conn.execute(
            "SELECT * FROM ideas WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return idea_row_to_dict(row)


app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")
