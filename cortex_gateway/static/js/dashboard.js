// dashboard.js — Dashboard page logic

// ── Auth guard ──
if (!requireAuth()) throw new Error('Not authenticated');

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ── API helpers ──
async function apiGet(url) {
    const res = await fetch(url, { headers: authHeaders() });
    if (res.status === 401) {
        clearToken();
        window.location.href = '/login';
        return null;
    }
    return res.json();
}

async function apiPost(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(body),
    });
    if (res.status === 401) {
        clearToken();
        window.location.href = '/login';
        return null;
    }
    return { ok: res.ok, data: await res.json() };
}

async function apiDelete(url) {
    const res = await fetch(url, {
        method: 'DELETE',
        headers: authHeaders(),
    });
    if (res.status === 401) {
        clearToken();
        window.location.href = '/login';
        return null;
    }
    return { ok: res.ok };
}

async function apiPut(url, body) {
    const res = await fetch(url, {
        method: 'PUT',
        headers: authHeaders(),
        body: JSON.stringify(body),
    });
    if (res.status === 401) {
        clearToken();
        window.location.href = '/login';
        return null;
    }
    return { ok: res.ok, data: await res.json() };
}

// ── Format helpers ──
function formatDate(ts) {
    return new Date(ts * 1000).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function formatNumber(n) {
    if (n == null) return '0';
    return n.toLocaleString();
}

function maskKey(key) {
    if (!key || key.length < 12) return key;
    return key.substring(0, 8) + '...' + key.substring(key.length - 4);
}

// ── API Keys ──
async function loadKeys() {
    const body = document.getElementById('keysTableBody');
    const data = await apiGet('/api/keys');
    if (!data) return;

    const keys = data.keys || [];
    if (keys.length === 0) {
        body.innerHTML = `
            <tr>
                <td colspan="5" class="empty-state">
                    <div class="icon">🔑</div>
                    <p>No API keys yet. Create one to get started.</p>
                </td>
            </tr>`;
        return;
    }

    body.innerHTML = keys.map(k => `
        <tr>
            <td><strong>${escapeHtml(k.name)}</strong></td>
            <td>
                <div class="key-display">
                    <code class="mono key-text" title="${escapeHtml(k.api_key)}">${maskKey(k.api_key)}</code>
                    <button class="copy-btn" onclick="copyKey('${escapeHtml(k.api_key)}', this)">Copy</button>
                </div>
            </td>
            <td>${formatDate(k.created_at)}</td>
            <td>
                ${k.active
                    ? '<span class="status-badge online"><span class="dot"></span> Active</span>'
                    : '<span class="status-badge offline"><span class="dot"></span> Revoked</span>'}
            </td>
            <td>
                ${k.active
                    ? `<button class="btn btn-danger btn-sm" onclick="revokeKey('${escapeHtml(k.api_key)}')">Revoke</button>`
                    : ''}
            </td>
        </tr>
    `).join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

async function copyKey(key, btn) {
    try {
        await navigator.clipboard.writeText(key);
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    } catch {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = key;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    }
}

async function revokeKey(key) {
    if (!confirm('Are you sure you want to revoke this API key? This cannot be undone.')) return;
    const result = await apiDelete(`/api/keys/${key}`);
    if (result?.ok) {
        loadKeys();
    }
}

// ── Create Key Modal ──
function showCreateKeyModal() {
    document.getElementById('createKeyModal').classList.add('show');
    document.getElementById('keyName').value = '';
    document.getElementById('keyName').focus();
}

function hideCreateKeyModal() {
    document.getElementById('createKeyModal').classList.remove('show');
}

async function createKey() {
    const name = document.getElementById('keyName').value.trim();
    if (!name) {
        document.getElementById('keyName').focus();
        return;
    }

    const btn = document.getElementById('createKeyBtn');
    btn.textContent = 'Creating...';
    btn.disabled = true;

    const result = await apiPost('/api/keys', { name });
    if (result?.ok) {
        hideCreateKeyModal();
        const alert = document.getElementById('keyAlert');
        alert.innerHTML = `Key created: <code class="mono">${escapeHtml(result.data.api_key)}</code> — Copy it now, it won't be shown in full again.`;
        alert.classList.add('show');
        setTimeout(() => alert.classList.remove('show'), 15000);
        loadKeys();
    }

    btn.textContent = 'Create';
    btn.disabled = false;
}

// Close modal on overlay click
document.getElementById('createKeyModal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) hideCreateKeyModal();
});

// ── Stats ──
async function loadStats() {
    const body = document.getElementById('statsTableBody');
    const data = await apiGet('/api/stats');
    if (!data) return;

    const stats = data.stats || [];
    if (stats.length === 0) {
        body.innerHTML = `
            <tr>
                <td colspan="5" class="empty-state">
                    <div class="icon">📊</div>
                    <p>No activity recorded yet.</p>
                </td>
            </tr>`;
        return;
    }

    body.innerHTML = stats.map(s => `
        <tr>
            <td><strong>${escapeHtml(s.client_name)}</strong><br>
                <span class="mono" style="font-size: 0.7rem; color: var(--text-muted);">${maskKey(s.client_key)}</span>
            </td>
            <td><code class="mono">${escapeHtml(s.model)}</code></td>
            <td>${formatNumber(s.request_count)}</td>
            <td>${formatNumber(s.total_input_tokens)}</td>
            <td>${formatNumber(s.total_output_tokens)}</td>
        </tr>
    `).join('');
}

// ── Change Password ──
document.getElementById('changePasswordForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const errorAlert = document.getElementById('pwErrorAlert');
    const successAlert = document.getElementById('pwSuccessAlert');
    errorAlert.classList.remove('show');
    successAlert.classList.remove('show');

    const oldPw = document.getElementById('oldPassword').value;
    const newPw = document.getElementById('newPassword').value;
    const confirmPw = document.getElementById('confirmPassword').value;

    if (newPw !== confirmPw) {
        errorAlert.textContent = 'New passwords do not match.';
        errorAlert.classList.add('show');
        return;
    }

    if (newPw.length < 4) {
        errorAlert.textContent = 'New password must be at least 4 characters.';
        errorAlert.classList.add('show');
        return;
    }

    const result = await apiPost('/api/change-password', {
        old_password: oldPw,
        new_password: newPw,
    });

    if (result?.ok) {
        successAlert.textContent = 'Password updated successfully!';
        successAlert.classList.add('show');
        document.getElementById('changePasswordForm').reset();
        setTimeout(() => successAlert.classList.remove('show'), 5000);
    } else {
        errorAlert.textContent = result?.data?.detail || 'Failed to update password.';
        errorAlert.classList.add('show');
    }
});

// ── Model Config ──
async function loadModels() {
    const data = await apiGet('/api/models');
    if (!data) return;

    const models = data.models || {};

    if (models.simple) {
        document.getElementById('simpleProvider').value = models.simple.provider || 'anthropic';
        document.getElementById('simpleModelName').value = models.simple.model_name || '';
        document.getElementById('simpleBaseUrl').value = models.simple.base_url || 'https://api.anthropic.com';
        document.getElementById('simpleCurrentKey').innerHTML =
            `Current key: <code class="mono">${escapeHtml(models.simple.api_key_masked || '(not set)')}</code>`;
    }

    if (models.complex) {
        document.getElementById('complexProvider').value = models.complex.provider || 'anthropic';
        document.getElementById('complexModelName').value = models.complex.model_name || '';
        document.getElementById('complexBaseUrl').value = models.complex.base_url || 'https://api.anthropic.com';
        document.getElementById('complexCurrentKey').innerHTML =
            `Current key: <code class="mono">${escapeHtml(models.complex.api_key_masked || '(not set)')}</code>`;
    }
}

async function saveModels() {
    const errorAlert = document.getElementById('modelErrorAlert');
    const successAlert = document.getElementById('modelSuccessAlert');
    errorAlert.classList.remove('show');
    successAlert.classList.remove('show');

    const simpleProvider  = document.getElementById('simpleProvider').value;
    const simpleModelName = document.getElementById('simpleModelName').value.trim();
    const simpleApiKey    = document.getElementById('simpleApiKey').value.trim();
    const simpleBaseUrl   = document.getElementById('simpleBaseUrl').value.trim();

    const complexProvider  = document.getElementById('complexProvider').value;
    const complexModelName = document.getElementById('complexModelName').value.trim();
    const complexApiKey    = document.getElementById('complexApiKey').value.trim();
    const complexBaseUrl   = document.getElementById('complexBaseUrl').value.trim();

    if (!simpleModelName || !complexModelName) {
        errorAlert.textContent = 'Model name is required for both tiers.';
        errorAlert.classList.add('show');
        return;
    }

    const btn = document.getElementById('saveModelsBtn');
    btn.textContent = 'Saving...';
    btn.disabled = true;

    const payload = {
        simple: {
            provider: simpleProvider,
            model_name: simpleModelName,
            api_key: simpleApiKey,
            base_url: simpleBaseUrl || (simpleProvider === 'openai' ? 'https://api.openai.com/v1' : 'https://api.anthropic.com'),
        },
        complex: {
            provider: complexProvider,
            model_name: complexModelName,
            api_key: complexApiKey,
            base_url: complexBaseUrl || (complexProvider === 'openai' ? 'https://api.openai.com/v1' : 'https://api.anthropic.com'),
        },
    };

    const result = await apiPut('/api/models', payload);

    if (result?.ok) {
        successAlert.textContent = 'Model configuration saved successfully!';
        successAlert.classList.add('show');
        // Clear API key fields and refresh masked display
        document.getElementById('simpleApiKey').value = '';
        document.getElementById('complexApiKey').value = '';
        loadModels();
        setTimeout(() => successAlert.classList.remove('show'), 5000);
    } else {
        errorAlert.textContent = result?.data?.detail || 'Failed to save model configuration.';
        errorAlert.classList.add('show');
    }

    btn.innerHTML = '💾 Save Model Configuration';
    btn.disabled = false;
}

// ── Initial load ──
loadKeys();
loadStats();
loadModels();
