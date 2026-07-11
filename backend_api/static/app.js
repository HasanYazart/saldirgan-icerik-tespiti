const state = { token: sessionStorage.getItem("access_token"), lastMessageId: null, role: null };
const $ = (id) => document.getElementById(id);

function notice(message = "") { $("notice").textContent = message; }
function setAuthenticated(value) {
  $("authPanel").classList.toggle("hidden", value);
  $("appPanel").classList.toggle("hidden", !value);
  if (!value) $("adminPanel").classList.add("hidden");
}
async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "İstek tamamlanamadı.");
  return data;
}
async function refreshSession() {
  if (!state.token) return setAuthenticated(false);
  try {
    const user = await api("/api/me/status");
    state.role = user.role;
    $("userInfo").textContent = `${user.username} · ${user.warning_count} uyarı`;
    setAuthenticated(true);
    $("adminPanel").classList.toggle("hidden", user.role !== "admin");
    if (user.role === "admin") await loadAdminQueues();
  } catch (_) {
    sessionStorage.removeItem("access_token"); state.token = null; setAuthenticated(false);
  }
}
async function authenticate(path, username, password, asForm = false) {
  const options = { method: "POST" };
  if (asForm) {
    options.headers = { "Content-Type": "application/x-www-form-urlencoded" };
    options.body = new URLSearchParams({ username, password });
  } else {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify({ username, password });
  }
  const result = await api(path, options);
  state.token = result.access_token;
  sessionStorage.setItem("access_token", state.token);
  notice(); await refreshSession();
}

$("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await authenticate("/api/auth/login", $("loginUsername").value, $("loginPassword").value, true); }
  catch (error) { notice(error.message); }
});
$("registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await authenticate("/api/auth/register", $("registerUsername").value, $("registerPassword").value); }
  catch (error) { notice(error.message); }
});
$("logoutButton").addEventListener("click", () => {
  state.token = null; state.role = null; sessionStorage.removeItem("access_token"); setAuthenticated(false); notice();
});
$("message").addEventListener("input", () => { $("counter").textContent = `${$("message").value.length} / 2000`; });
$("analyzeForm").addEventListener("submit", async (event) => {
  event.preventDefault(); const button = event.submitter; button.disabled = true; notice();
  try {
    const data = await api("/api/chat/send", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: $("message").value })
    });
    const result = $("result"); result.className = `result ${data.status}`;
    result.textContent = `${data.status.toUpperCase()} · Skor %${(data.toxicity_score * 100).toFixed(1)} · ${data.displayed_text}`;
    state.lastMessageId = data.message_id;
    $("appealForm").classList.toggle("hidden", data.action_taken === "passed");
    await refreshSession();
  } catch (error) { notice(error.message); }
  finally { button.disabled = false; }
});
$("appealForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/appeals", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_id: state.lastMessageId, reason: $("appealReason").value })
    });
    $("appealForm").classList.add("hidden"); notice("İtirazınız inceleme kuyruğuna alındı.");
  } catch (error) { notice(error.message); }
});

function queueItem(title, detail, actions) {
  const item = document.createElement("article"); item.className = "queue-item";
  const heading = document.createElement("strong"); heading.textContent = title;
  const text = document.createElement("p"); text.textContent = detail;
  const buttons = document.createElement("div"); buttons.className = "queue-actions";
  actions.forEach(([label, handler]) => {
    const button = document.createElement("button"); button.textContent = label;
    button.addEventListener("click", handler); buttons.append(button);
  });
  item.append(heading, text, buttons); return item;
}
async function decideReview(id, decision) {
  const note = window.prompt("Karar notu:"); if (!note) return;
  await api(`/api/admin/reviews/${id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, admin_note: note }) });
  await loadAdminQueues();
}
async function decideAppeal(id, decision) {
  const note = window.prompt("Karar notu:"); if (!note) return;
  await api(`/api/admin/appeals/${id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, admin_note: note }) });
  await loadAdminQueues();
}
async function loadAdminQueues() {
  if (state.role !== "admin") return;
  try {
    const [reviews, appeals] = await Promise.all([api("/api/admin/reviews"), api("/api/admin/appeals")]);
    const reviewQueue = $("reviewQueue"); reviewQueue.replaceChildren();
    reviews.forEach((record) => reviewQueue.append(queueItem(
      `#${record.id} · %${(record.toxicity_score * 100).toFixed(1)}`,
      record.content || "İçerik çözülemedi",
      [["Onayla", () => decideReview(record.id, "approved")], ["Geçersiz say", () => decideReview(record.id, "rejected")]]
    )));
    if (!reviews.length) reviewQueue.textContent = "Bekleyen kayıt yok.";
    const appealQueue = $("appealQueue"); appealQueue.replaceChildren();
    appeals.forEach((appeal) => appealQueue.append(queueItem(
      `#${appeal.id} · Mesaj ${appeal.message_id}`, appeal.reason,
      [["Kabul et", () => decideAppeal(appeal.id, "accepted")], ["Reddet", () => decideAppeal(appeal.id, "rejected")]]
    )));
    if (!appeals.length) appealQueue.textContent = "Açık itiraz yok.";
  } catch (error) { notice(error.message); }
}
$("refreshAdmin").addEventListener("click", loadAdminQueues);

api("/api/health").then((health) => {
  const chip = $("health"); chip.textContent = health.ready ? `BERT aktif · ${health.model_version}` : "Kural modu · model kullanılamıyor";
  chip.classList.add(health.ready ? "ok" : "warn");
}).catch(() => { $("health").textContent = "Sistem erişilemiyor"; });
refreshSession();
