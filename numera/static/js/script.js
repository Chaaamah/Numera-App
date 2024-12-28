document.addEventListener("DOMContentLoaded", () => {
    const themeToggle = document.querySelector("#theme-toggle");
    themeToggle?.addEventListener("click", () => {
        document.body.classList.toggle("dark-mode");
    });
});
