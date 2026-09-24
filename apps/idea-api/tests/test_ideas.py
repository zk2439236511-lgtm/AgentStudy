import sqlite3

import pytest
from fastapi.testclient import TestClient

from main import DB_PATH, app

client = TestClient(app)

SEED_TITLE = "__pytest_seed_idea__"
SEED_DESC = "pytest 播种的测试数据"


@pytest.fixture()
def seed_idea():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO ideas (title, "desc", created_at)
           VALUES (?, ?, datetime('now', 'localtime'))""",
        (SEED_TITLE, SEED_DESC),
    )
    conn.commit()
    seed_id = cursor.lastrowid
    conn.close()
    yield seed_id
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM ideas WHERE id = ?", (seed_id,))
    conn.commit()
    conn.close()


def test_get_ideas_status_is_200():
    response = client.get("/ideas")
    assert response.status_code == 200


def test_get_ideas_returns_list():
    data = client.get("/ideas").json()
    assert isinstance(data, list)


def test_get_ideas_items_have_title_and_desc(seed_idea):
    data = client.get("/ideas").json()
    item = next(idea for idea in data if idea["id"] == seed_idea)
    assert "title" in item
    assert "desc" in item
    assert item["title"] == SEED_TITLE
    assert item["desc"] == SEED_DESC


CREATED_PREFIX = "__pytest_created_idea__"


@pytest.fixture()
def cleanup_created_ideas():
    yield
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM ideas WHERE title LIKE ?", (f"{CREATED_PREFIX}%",))
    conn.commit()
    conn.close()


def test_post_ideas_status_is_201(cleanup_created_ideas):
    response = client.post(
        "/ideas",
        json={"title": f"{CREATED_PREFIX}状态码", "desc": "验证创建成功返回 201"},
    )
    assert response.status_code == 201


def test_post_ideas_echoes_title_and_desc(cleanup_created_ideas):
    title = f"{CREATED_PREFIX}回显"
    desc = "响应里应原样带回标题和描述"
    data = client.post("/ideas", json={"title": title, "desc": desc}).json()
    assert data["title"] == title
    assert data["desc"] == desc
    assert isinstance(data["id"], int)
    assert data["created_at"]


def test_post_ideas_persists_to_list(cleanup_created_ideas):
    created = client.post(
        "/ideas",
        json={"title": f"{CREATED_PREFIX}落库", "desc": "创建后应能在列表里查到"},
    ).json()
    data = client.get("/ideas").json()
    item = next(idea for idea in data if idea["id"] == created["id"])
    assert item["title"] == created["title"]
    assert item["desc"] == created["desc"]


def test_post_ideas_blank_title_returns_400(cleanup_created_ideas):
    response = client.post("/ideas", json={"title": "   ", "desc": "标题只有空格"})
    assert response.status_code == 400


def test_post_ideas_missing_desc_returns_422(cleanup_created_ideas):
    response = client.post("/ideas", json={"title": f"{CREATED_PREFIX}缺字段"})
    assert response.status_code == 422
