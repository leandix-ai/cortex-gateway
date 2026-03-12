// home.js — Public home page logic

(async function () {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const gatewayStatus = document.getElementById('gatewayStatus');
    const totalRequests = document.getElementById('totalRequests');
    const uptimeEl = document.getElementById('uptime');
    const simpleModel = document.getElementById('simpleModel');
    const complexModel = document.getElementById('complexModel');

    function formatUptime(seconds) {
        const d = Math.floor(seconds / 86400);
        const h = Math.floor((seconds % 86400) / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        if (d > 0) return `${d}d ${h}h`;
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m`;
    }

    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        if (data.status === 'online') {
            statusDot.style.background = '#10b981';
            statusText.textContent = 'Online';
            gatewayStatus.innerHTML = '<span class="status-badge online"><span class="dot"></span> Online</span>';
        } else {
            statusDot.style.background = '#f43f5e';
            statusText.textContent = 'Offline';
            gatewayStatus.innerHTML = '<span class="status-badge offline"><span class="dot"></span> Offline</span>';
        }

        totalRequests.textContent = data.total_requests?.toLocaleString() || '0';
        uptimeEl.textContent = formatUptime(data.uptime_seconds || 0);
        simpleModel.textContent = data.models?.simple || '—';
        complexModel.textContent = data.models?.complex || '—';
    } catch (e) {
        statusDot.style.background = '#f43f5e';
        statusText.textContent = 'Unreachable';
        gatewayStatus.innerHTML = '<span class="status-badge offline"><span class="dot"></span> Unreachable</span>';
        totalRequests.textContent = '—';
        uptimeEl.textContent = '—';
        simpleModel.textContent = '—';
        complexModel.textContent = '—';
    }
})();
