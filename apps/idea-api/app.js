const API = "/api/todos";

const composer = document.getElementById("composer");
const input = document.getElementById("todo-input");
const listEl = document.getElementById("todo-list");
const countEl = document.getElementById("count");
const emptyEl = document.getElementById("empty");
const filtersEl = document.getElementById("filters");
const clearDoneBtn = document.getElementById("clear-done");
const todayEl = document.getElementById("today");

let todos = [];
let filter = "all";

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `请求失败（${response.status}）`);
  }
  return response.status === 204 ? null : response.json();
}

function showError(error) {
  console.error(error);
  window.alert(error.message);
}

async function loadTodos() {
  try {
    todos = await request(API);
    render();
  } catch (error) {
    showError(error);
  }
}

async function addTodo(content) {
  await request(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  await loadTodos();
}

async function patchTodo(id, changes) {
  await request(`${API}/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  await loadTodos();
}

async function removeTodo(id) {
  await request(`${API}/${id}`, { method: "DELETE" });
  await loadTodos();
}

function visibleTodos() {
  if (filter === "active") return todos.filter((item) => !item.done);
  if (filter === "done") return todos.filter((item) => item.done);
  return todos;
}

function createItem(todo) {
  const li = document.createElement("li");
  li.className = "item" + (todo.done ? " is-done" : "");

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = "item__check";
  checkbox.checked = todo.done;
  checkbox.addEventListener("change", () => {
    patchTodo(todo.id, { done: checkbox.checked }).catch(showError);
  });

  const text = document.createElement("span");
  text.className = "item__text";
  text.textContent = todo.content;

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "item__delete";
  removeBtn.textContent = "✕";
  removeBtn.title = "删除";
  removeBtn.addEventListener("click", () => {
    removeTodo(todo.id).catch(showError);
  });

  li.append(checkbox, text, removeBtn);
  return li;
}

function render() {
  const items = visibleTodos();
  listEl.replaceChildren(...items.map(createItem));

  const doneCount = todos.filter((item) => item.done).length;
  countEl.textContent = `共 ${todos.length} 项，已完成 ${doneCount} 项`;

  emptyEl.textContent =
    todos.length === 0 ? "还没有待办事项，先加一条吧。" : "当前筛选条件下没有内容。";
  emptyEl.classList.toggle("is-hidden", items.length > 0);
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content) return;
  input.value = "";
  try {
    await addTodo(content);
  } catch (error) {
    showError(error);
  }
  input.focus();
});

filtersEl.addEventListener("click", (event) => {
  const button = event.target.closest(".filter");
  if (!button) return;
  filter = button.dataset.filter;
  for (const item of filtersEl.querySelectorAll(".filter")) {
    item.classList.toggle("is-active", item === button);
  }
  render();
});

clearDoneBtn.addEventListener("click", async () => {
  try {
    const doneItems = todos.filter((item) => item.done);
    for (const item of doneItems) {
      await request(`${API}/${item.id}`, { method: "DELETE" });
    }
    await loadTodos();
  } catch (error) {
    showError(error);
  }
});

todayEl.textContent = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "long",
  day: "numeric",
  weekday: "long",
}).format(new Date());

loadTodos();
