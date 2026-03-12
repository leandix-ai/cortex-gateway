// auth.js — Login page logic + shared auth utilities

const AUTH_TOKEN_KEY = 'cortex_admin_token';

// ── Shared auth utilities ──

function getToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setToken(token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function clearToken() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
}

function authHeaders() {
    const token = getToken();
    return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } : {};
}

function requireAuth() {
    if (!getToken()) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

// ── Login form handler ──

const loginForm = document.getElementById('loginForm');
if (loginForm) {
    // If already logged in, redirect to dashboard
    if (getToken()) {
        window.location.href = '/dashboard';
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const errorAlert = document.getElementById('errorAlert');
        const loginBtn = document.getElementById('loginBtn');
        const password = document.getElementById('password').value;

        errorAlert.classList.remove('show');
        loginBtn.textContent = 'Signing in...';
        loginBtn.disabled = true;

        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password }),
            });

            if (res.ok) {
                const data = await res.json();
                setToken(data.token);
                window.location.href = '/dashboard';
            } else {
                errorAlert.textContent = 'Invalid password. Please try again.';
                errorAlert.classList.add('show');
            }
        } catch (err) {
            errorAlert.textContent = 'Connection error. Please try again.';
            errorAlert.classList.add('show');
        }

        loginBtn.textContent = 'Sign In';
        loginBtn.disabled = false;
    });
}
