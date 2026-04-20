+++
title = '공사장의 구석'
date = 2024-09-24T00:00:01+09:00
draft = false
disable_comments = true
+++

<div id="gol-controls" style="margin-bottom: 8px; display: flex; gap: 8px; align-items: center;">
    <button id="gol-toggle">Pause</button>
    <button id="gol-reset">Reset</button>
    <span id="gol-count" style="opacity: 0.6;">alive: 0</span>
    <span id="gol-gen" style="opacity: 0.6;">gen: 0</span>
</div>
<pre id="gol"></pre>
<p style="opacity: 0.4; font-size: 0.85em;">click to toggle cells</p>

<style>
    #gol {
        font-size: 10px;
        line-height: 1.1;
        letter-spacing: 1px;
        user-select: none;
        background: inherit;
        color: inherit;
    }
    #gol-controls button {
        font-family: inherit;
        font-size: inherit;
        background: transparent;
        color: inherit;
        border: 1px solid;
        padding: 2px 8px;
        cursor: pointer;
    }
</style>

<script>
(function() {
    var W = 60, H = 30;
    var grid = [];
    var running = true;
    var generation = 0;
    var el = document.getElementById('gol');
    var countEl = document.getElementById('gol-count');
    var genEl = document.getElementById('gol-gen');
    var toggleBtn = document.getElementById('gol-toggle');
    var resetBtn = document.getElementById('gol-reset');
    var timer = null;

    function init() {
        grid = [];
        generation = 0;
        for (var y = 0; y < H; y++) {
            grid[y] = [];
            for (var x = 0; x < W; x++) {
                grid[y][x] = Math.random() < 0.3 ? 1 : 0;
            }
        }
    }

    function countNeighbors(g, x, y) {
        var c = 0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                var ny = (y + dy + H) % H;
                var nx = (x + dx + W) % W;
                c += g[ny][nx];
            }
        }
        return c;
    }

    function step() {
        var next = [];
        for (var y = 0; y < H; y++) {
            next[y] = [];
            for (var x = 0; x < W; x++) {
                var n = countNeighbors(grid, x, y);
                if (grid[y][x]) {
                    next[y][x] = (n === 2 || n === 3) ? 1 : 0;
                } else {
                    next[y][x] = (n === 3) ? 1 : 0;
                }
            }
        }
        grid = next;
        generation++;
    }

    function render() {
        var alive = 0;
        var lines = [];
        for (var y = 0; y < H; y++) {
            var row = '';
            for (var x = 0; x < W; x++) {
                if (grid[y][x]) {
                    row += '#';
                    alive++;
                } else {
                    row += '.';
                }
            }
            lines.push(row);
        }
        el.textContent = lines.join('\n');
        countEl.textContent = 'alive: ' + alive;
        genEl.textContent = 'gen: ' + generation;
    }

    function tick() {
        step();
        render();
    }

    function start() {
        if (!timer) timer = setInterval(tick, 150);
    }

    function stop() {
        clearInterval(timer);
        timer = null;
    }

    toggleBtn.addEventListener('click', function() {
        if (running) {
            stop();
            toggleBtn.textContent = 'Play';
        } else {
            start();
            toggleBtn.textContent = 'Pause';
        }
        running = !running;
    });

    resetBtn.addEventListener('click', function() {
        init();
        generation = 0;
        render();
    });

    el.addEventListener('click', function(e) {
        var rect = el.getBoundingClientRect();
        var style = window.getComputedStyle(el);
        var charW = el.scrollWidth / (W + 1);
        var charH = el.scrollHeight / H;
        var mx = e.clientX - rect.left;
        var my = e.clientY - rect.top;
        var cx = Math.floor(mx / charW);
        var cy = Math.floor(my / charH);
        if (cx >= 0 && cx < W && cy >= 0 && cy < H) {
            grid[cy][cx] = grid[cy][cx] ? 0 : 1;
            render();
        }
    });

    init();
    render();
    start();
})();
</script>
