// Force Documenter docs to always use the light theme.
// This overrides the default Documenter themeswap behavior.
(function force_light_theme() {
  var forcedTheme = "documenter-light";

  try {
    if (window.localStorage != null) {
      window.localStorage.setItem("documenter-theme", forcedTheme);
    }
  } catch (_err) {
    // Ignore localStorage errors (for example in private mode).
  }

  var html = document.getElementsByTagName("html")[0];
  if (html) {
    // documenter-light is the primary light theme; empty class matches the default behavior.
    html.className = "";
  }

  for (var i = 0; i < document.styleSheets.length; i++) {
    var ss = document.styleSheets[i];
    if (!ss || !ss.ownerNode || !ss.ownerNode.getAttribute) continue;
    var themeName = ss.ownerNode.getAttribute("data-theme-name");
    if (themeName === null) continue;
    ss.disabled = themeName !== forcedTheme;
  }
})();
