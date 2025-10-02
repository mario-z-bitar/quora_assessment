const AGGREGATE_PATH = "../outputs/aggregate_summary.json";
const PER_CONVO_PATH = "../outputs/per_conversation_metrics.json";

const summaryHost = document.querySelector("#summary-cards");
const conversationTypeHost = document.querySelector("#conversation-type");
const intentHost = document.querySelector("#intent-distribution");
const topTopicsHost = document.querySelector("#top-topics");
const actionItemsHost = document.querySelector("#action-items");
const tableBody = document.querySelector("#conversation-table tbody");
const filterInput = document.querySelector("#table-filter");
const reloadButton = document.querySelector("#reload");

let conversationRowsCache = [];

async function loadData() {
  try {
    toggleLoading(true);
    const [aggregate, perConversation] = await Promise.all([
      fetchJson(AGGREGATE_PATH),
      fetchJson(PER_CONVO_PATH),
    ]);
    renderDashboard(aggregate, perConversation);
  } catch (error) {
    showError(error);
  } finally {
    toggleLoading(false);
  }
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.json();
}

function renderDashboard(aggregate, perConversation) {
  renderSummary(aggregate);
  renderDistribution(conversationTypeHost, "Conversation type", aggregate.conversation_type_distribution);
  renderDistribution(intentHost, "Intent", aggregate.intent_distribution);
  renderTopTopics(aggregate.top_topics);
  renderActionItems(aggregate.action_items);
  renderTable(perConversation.conversations || []);
}

function renderSummary(aggregate) {
  summaryHost.innerHTML = "";
  const cards = [
    {
      title: "Total conversations",
      value: formatNumber(aggregate.total_conversations),
      caption: `Generated ${formatTimestamp(aggregate.generated_at)}`,
    },
    {
      title: "Avg. messages",
      value: formatNumber(aggregate.average_messages, 1),
      caption: "Across processed conversations",
    },
    {
      title: "Avg. characters",
      value: formatNumber(aggregate.average_character_count, 0),
      caption: "Total characters per conversation",
    },
    {
      title: "Median duration",
      value: formatDuration(aggregate.median_duration_seconds),
      caption: "Conversation span (seconds)",
    },
  ];

  for (const card of cards) {
    const template = document.querySelector("#summary-card-template");
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector("h3").textContent = card.title;
    node.querySelector(".value").textContent = card.value;
    node.querySelector(".caption").textContent = card.caption;
    summaryHost.appendChild(node);
  }
}

function renderDistribution(container, title, distribution = {}) {
  container.innerHTML = "";
  const heading = document.createElement("h3");
  heading.textContent = title;
  container.appendChild(heading);

  const total = Object.values(distribution).reduce((sum, value) => sum + value, 0);
  const list = document.createElement("div");
  list.className = "bar-list";

  if (!total) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No data available";
    container.appendChild(empty);
    return;
  }

  Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, count]) => {
      const percent = total ? Math.round((count / total) * 100) : 0;
      const row = document.createElement("div");
      row.className = "bar-row";

      const labelEl = document.createElement("span");
      labelEl.textContent = prettify(label);
      row.appendChild(labelEl);

      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${percent}%`;
      bar.appendChild(fill);
      row.appendChild(bar);

      const valueEl = document.createElement("span");
      valueEl.textContent = `${count} (${percent}%)`;
      valueEl.className = "muted";
      row.appendChild(valueEl);

      list.appendChild(row);
    });

  container.appendChild(list);
}

function renderTopTopics(topics = []) {
  topTopicsHost.innerHTML = "";
  if (!topics.length) {
    const li = document.createElement("li");
    li.textContent = "No topics found.";
    topTopicsHost.appendChild(li);
    return;
  }

  topics.forEach(([topic, count]) => {
    const li = document.createElement("li");
    const label = document.createElement("strong");
    label.textContent = prettify(topic);
    li.appendChild(label);
    const span = document.createElement("span");
    span.className = "muted";
    span.textContent = ` – ${count} conversation${count === 1 ? "" : "s"}`;
    li.appendChild(span);
    topTopicsHost.appendChild(li);
  });
}

function renderActionItems(items = []) {
  actionItemsHost.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No action items suggested.";
    actionItemsHost.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    actionItemsHost.appendChild(li);
  });
}

function renderTable(conversations) {
  tableBody.innerHTML = "";
  conversationRowsCache = conversations.map((conversation) => {
    const row = document.createElement("tr");
    row.dataset.topic = conversation.topic || "";
    row.dataset.intent = conversation.conversation_intent || "";

    const cells = [
      conversation.conversation_id,
      prettify(conversation.topic),
      prettify(conversation.conversation_intent),
      prettify(conversation.conversation_type),
      formatNumber(conversation.message_count),
      formatNumber(conversation.total_character_count),
      `${Math.round((conversation.topic_confidence || 0) * 100)}%`,
    ];

    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "";
      row.appendChild(cell);
    });

    tableBody.appendChild(row);
    return row;
  });
}

function formatNumber(value, decimals = 0) {
  if (value === undefined || value === null) return "–";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatTimestamp(value) {
  if (!value) return "unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  const totalSeconds = Number(seconds);
  if (!Number.isFinite(totalSeconds)) return "–";
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = Math.round(totalSeconds % 60);
  if (minutes === 0) return `${remainingSeconds}s`;
  return `${minutes}m ${remainingSeconds}s`;
}

function prettify(text = "") {
  return text
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function toggleLoading(isLoading) {
  document.body.classList.toggle("loading", isLoading);
  reloadButton.disabled = isLoading;
}

function showError(error) {
  console.error(error);
  const bannerId = "error-banner";
  let banner = document.getElementById(bannerId);
  if (!banner) {
    banner = document.createElement("div");
    banner.id = bannerId;
    banner.className = "error-banner";
    document.body.prepend(banner);
  }
  banner.textContent = error.message || "Failed to load dashboard data.";
}

function applyFilter(term) {
  const normalized = term.trim().toLowerCase();
  conversationRowsCache.forEach((row) => {
    if (!normalized) {
      row.hidden = false;
      return;
    }
    const haystack = `${row.dataset.topic} ${row.dataset.intent}`.toLowerCase();
    row.hidden = !haystack.includes(normalized);
  });
}

filterInput.addEventListener("input", (event) => {
  applyFilter(event.target.value);
});

reloadButton.addEventListener("click", () => {
  loadData();
});

loadData();
