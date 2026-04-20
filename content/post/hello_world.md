+++
title = 'Hello World'
date = 2024-09-24T00:00:00+09:00
draft = false
disable_comments = true
+++

<style>
    .ascii-art {
        font-size: 10px;
        line-height: 1;
        letter-spacing: 2px;
        text-align: center;
        background: inherit;
        color: inherit;
    }
    .shape-label {
        text-align: center;
        opacity: 0.4;
        margin: 0 0 2em 0;
        font-size: 0.85em;
    }
</style>

<pre id="donut" class="ascii-art"></pre>
<p class="shape-label">Torus</p>

<pre id="cube" class="ascii-art"></pre>
<p class="shape-label">Cube</p>

<pre id="tetra" class="ascii-art"></pre>
<p class="shape-label">Tetrahedron</p>

<pre id="sphere" class="ascii-art"></pre>
<p class="shape-label">Sphere</p>

<script>
(function() {
    var A = 1, B = 1;
    var el = document.getElementById('donut');

    function render() {
        var b = [];
        var z = [];
        var width = 80, height = 30;
        for (var k = 0; k < width * height; k++) {
            b[k] = k % width === width - 1 ? '\n' : ' ';
            z[k] = 0;
        }

        for (var j = 0; j < 6.28; j += 0.07) {
            for (var i = 0; i < 6.28; i += 0.02) {
                var c = Math.sin(i),
                    d = Math.cos(j),
                    e = Math.sin(A),
                    f = Math.sin(j),
                    g = Math.cos(A),
                    h = d + 2,
                    D = 1 / (c * h * e + f * g + 5),
                    l = Math.cos(i),
                    m = Math.cos(B),
                    n = Math.sin(B),
                    t = c * h * g - f * e;

                var x = (40 + 30 * D * (l * h * m - t * n)) | 0;
                var y = (15 + 15 * D * (l * h * n + t * m)) | 0;
                var o = x + width * y;
                var N = (8 * ((f * e - c * d * g) * m - c * d * e - f * g - l * d * n)) | 0;

                if (y > 0 && y < height && x > 0 && x < width && D > z[o]) {
                    z[o] = D;
                    b[o] = '.,-~:;=!*#$@'[N > 0 ? N : 0];
                }
            }
        }

        el.textContent = b.join('');
        A += 0.04;
        B += 0.02;
    }

    setInterval(render, 50);
})();
</script>

<script>
(function() {
    var A = 0, B = 0, C = 0;
    var el = document.getElementById('cube');
    var width = 80, height = 30;
    var cubeWidth = 20;
    var bg = ' ';
    var distFromCam = 60;
    var K1 = 40;

    function project(x, y, z) {
        var ooz = 1 / (z + distFromCam);
        var xp = (width / 2 + K1 * ooz * x) | 0;
        var yp = (height / 2 - K1 * ooz * y) | 0;
        return { x: xp, y: yp, ooz: ooz };
    }

    function rotate(x, y, z) {
        var sa = Math.sin(A), ca = Math.cos(A);
        var sb = Math.sin(B), cb = Math.cos(B);
        var sc = Math.sin(C), cc = Math.cos(C);
        var y1 = y * ca - z * sa, z1 = y * sa + z * ca;
        var x2 = x * cb + z1 * sb, z2 = -x * sb + z1 * cb;
        var x3 = x2 * cc - y1 * sc, y3 = x2 * sc + y1 * cc;
        return { x: x3, y: y3, z: z2 };
    }

    function render() {
        var buf = [];
        var zbuf = [];
        for (var k = 0; k < width * height; k++) {
            buf[k] = k % width === width - 1 ? '\n' : bg;
            zbuf[k] = 0;
        }
        var half = cubeWidth / 2;
        var step = 0.6;
        var faces = [
            { gen: function(a, b) { return { x:  half, y: a, z: b }; }, ch: '@' },
            { gen: function(a, b) { return { x: -half, y: a, z: b }; }, ch: '#' },
            { gen: function(a, b) { return { x: a, y:  half, z: b }; }, ch: '~' },
            { gen: function(a, b) { return { x: a, y: -half, z: b }; }, ch: '=' },
            { gen: function(a, b) { return { x: a, y: b, z:  half }; }, ch: ';' },
            { gen: function(a, b) { return { x: a, y: b, z: -half }; }, ch: '+' }
        ];
        for (var f = 0; f < faces.length; f++) {
            for (var a = -half; a < half; a += step) {
                for (var b = -half; b < half; b += step) {
                    var p = faces[f].gen(a, b);
                    var r = rotate(p.x, p.y, p.z);
                    var proj = project(r.x, r.y, r.z);
                    var idx = proj.x + proj.y * width;
                    if (proj.x >= 0 && proj.x < width - 1 && proj.y >= 0 && proj.y < height) {
                        if (proj.ooz > zbuf[idx]) {
                            zbuf[idx] = proj.ooz;
                            buf[idx] = faces[f].ch;
                        }
                    }
                }
            }
        }
        el.textContent = buf.join('');
        A += 0.03;
        B += 0.04;
        C += 0.02;
    }
    setInterval(render, 50);
})();
</script>

<script>
(function() {
    var A = 0, B = 0, C = 0;
    var el = document.getElementById('tetra');
    var width = 80, height = 30;
    var distFromCam = 50;
    var K1 = 40;
    var s = 15;

    // tetrahedron vertices
    var V = [
        { x: 0,          y:  s,          z: 0 },
        { x:  s * 0.943, y: -s / 3,      z: 0 },
        { x: -s * 0.471, y: -s / 3,      z:  s * 0.816 },
        { x: -s * 0.471, y: -s / 3,      z: -s * 0.816 }
    ];
    var faces = [
        { v: [0, 1, 2], ch: '@' },
        { v: [0, 2, 3], ch: '#' },
        { v: [0, 3, 1], ch: '=' },
        { v: [1, 3, 2], ch: '~' }
    ];

    function rotate(p) {
        var sa = Math.sin(A), ca = Math.cos(A);
        var sb = Math.sin(B), cb = Math.cos(B);
        var sc = Math.sin(C), cc = Math.cos(C);
        var y1 = p.y * ca - p.z * sa, z1 = p.y * sa + p.z * ca;
        var x2 = p.x * cb + z1 * sb, z2 = -p.x * sb + z1 * cb;
        var x3 = x2 * cc - y1 * sc, y3 = x2 * sc + y1 * cc;
        return { x: x3, y: y3, z: z2 };
    }

    function render() {
        var buf = [];
        var zbuf = [];
        for (var k = 0; k < width * height; k++) {
            buf[k] = k % width === width - 1 ? '\n' : ' ';
            zbuf[k] = 0;
        }

        for (var f = 0; f < faces.length; f++) {
            var v0 = V[faces[f].v[0]], v1 = V[faces[f].v[1]], v2 = V[faces[f].v[2]];
            for (var u = 0; u <= 1; u += 0.02) {
                for (var v = 0; v <= 1 - u; v += 0.02) {
                    var w = 1 - u - v;
                    var px = v0.x * u + v1.x * v + v2.x * w;
                    var py = v0.y * u + v1.y * v + v2.y * w;
                    var pz = v0.z * u + v1.z * v + v2.z * w;
                    var r = rotate({ x: px, y: py, z: pz });
                    var ooz = 1 / (r.z + distFromCam);
                    var xp = (width / 2 + K1 * ooz * r.x) | 0;
                    var yp = (height / 2 - K1 * ooz * r.y) | 0;
                    var idx = xp + yp * width;
                    if (xp >= 0 && xp < width - 1 && yp >= 0 && yp < height && ooz > zbuf[idx]) {
                        zbuf[idx] = ooz;
                        buf[idx] = faces[f].ch;
                    }
                }
            }
        }

        el.textContent = buf.join('');
        A += 0.03;
        B += 0.02;
        C += 0.04;
    }
    setInterval(render, 50);
})();
</script>

<script>
(function() {
    var A = 0, B = 0;
    var el = document.getElementById('sphere');
    var width = 80, height = 30;
    var R = 10;

    function render() {
        var buf = [];
        var zbuf = [];
        for (var k = 0; k < width * height; k++) {
            buf[k] = k % width === width - 1 ? '\n' : ' ';
            zbuf[k] = 0;
        }
        var sa = Math.sin(A), ca = Math.cos(A);
        var sb = Math.sin(B), cb = Math.cos(B);

        for (var phi = 0; phi < 6.28; phi += 0.07) {
            for (var theta = 0; theta < 6.28; theta += 0.02) {
                var sp = Math.sin(phi), cp = Math.cos(phi);
                var st = Math.sin(theta), ct = Math.cos(theta);
                var x0 = R * sp * ct;
                var y0 = R * sp * st;
                var z0 = R * cp;
                var y1 = y0 * ca - z0 * sa;
                var z1 = y0 * sa + z0 * ca;
                var x2 = x0 * cb + z1 * sb;
                var z2 = -x0 * sb + z1 * cb;
                var ooz = 1 / (z2 + 30);
                var xp = (40 + 25 * ooz * x2) | 0;
                var yp = (15 + 17 * ooz * y1) | 0;
                var idx = xp + yp * width;
                var L = (sp * ct * cb - sp * st * sa * sb + cp * ca * sb) * 2
                       + (sp * ct * sb + sp * st * sa * cb - cp * ca * cb);
                var li = (L * 4) | 0;
                if (xp >= 0 && xp < width - 1 && yp >= 0 && yp < height && ooz > zbuf[idx]) {
                    zbuf[idx] = ooz;
                    buf[idx] = '.,:;ox%#@'[li > 0 ? (li < 9 ? li : 8) : 0];
                }
            }
        }
        el.textContent = buf.join('');
        A += 0.03;
        B += 0.02;
    }
    setInterval(render, 50);
})();
</script>
