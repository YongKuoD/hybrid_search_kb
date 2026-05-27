/**
 * 混合检索知识库 — 前端共用工具
 * API 错误解析 / Toast / 按钮忙碌态 / XSS 转义
 */
(function (g) {
  "use strict";

  function parseDetail(json) {
    if (!json || json.detail == null) return "";
    var d = json.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map(function (e) {
          if (!e) return "";
          if (typeof e.msg === "string") return e.msg;
          return JSON.stringify(e);
        })
        .filter(Boolean)
        .join(" · ");
    }
    if (typeof d === "object") {
      try { return JSON.stringify(d); }
      catch (e) { return String(d); }
    }
    return String(d);
  }

  async function readError(res) {
    var t = await res.text();
    try {
      var j = JSON.parse(t);
      return parseDetail(j) || res.statusText;
    } catch (e) {
      return t || res.statusText;
    }
  }

  async function reqJson(res) {
    var t = await res.text();
    var j = {};
    try { j = t ? JSON.parse(t) : {}; }
    catch (e) { j = {}; }
    if (!res.ok) throw new Error(parseDetail(j) || res.statusText || String(res.status));
    return j;
  }

  function toast(msg, kind) {
    kind = kind || "info";
    var el = document.getElementById("toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "toast";
      el.setAttribute("role", "status");
      el.setAttribute("aria-live", "polite");
      document.body.appendChild(el);
    }
    el.className = "toast toast-" + kind;
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(function () { el.hidden = true; }, 4200);
  }

  function busy(btn, on) {
    if (!btn) return;
    btn.disabled = !!on;
    btn.setAttribute("aria-busy", on ? "true" : "false");
  }

  function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function escAttr(s) {
    return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  g.KB = {
    parseDetail: parseDetail,
    readError: readError,
    reqJson: reqJson,
    toast: toast,
    busy: busy,
    escHtml: escHtml,
    escAttr: escAttr,
  };
})(typeof window !== "undefined" ? window : this);
