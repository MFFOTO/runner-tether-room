# -*- coding: utf-8 -*-
"""One look for Tether Room and Grade Room.

Both pages are single HTML strings that style themselves with the same colour
variables. This module owns those variables for a light theme (the default) and
a dark one, plus a short layer of shared component polish that is appended after
each page's own CSS, and the Light/Dark switch. A page opts in by carrying these
placeholders, which apply() fills in:

    /*@THEME_TOKENS@*/   where the :root colour block used to be
    /*@THEME_POLISH@*/   at the end of the <style> block
    <!--@THEME_HEAD@-->  inside <head>: sets the theme before first paint
    <!--@THEME_TOGGLE@--> in the header: the Light / Dark switch
    /*@THEME_JS@*/       in the page script

The choice (one icon button in the header) is remembered in the browser (localStorage) and the page works
without it. Photos are always shown on a dark surround in the lightbox, because
that is how they are judged.
"""

from __future__ import annotations

TOKENS = r"""
:root{color-scheme:light;
 --ground:#f3f5f8;--panel:#ffffff;--panel-2:#f6f8fb;--panel-3:#e9eef4;
 --line:#e1e6ed;--line-hi:#cbd3de;--ink:#101828;--muted:#566377;--dim:#5f6d80;
 --accent:#0f766e;--accent-hi:#115e59;--accent-ink:#ffffff;--accent-dim:#9fd5cd;--accent-soft:#e3f4f1;
 --live:#0f9d8a;--live-soft:#e3f4f1;--live-line:#a7dcd3;
 --bad:#d92d20;--bad-soft:#fdecea;--bad-line:#f2b8b3;
 --warn:#b45309;--warn-soft:#fef3c7;--warn-line:#f3d98a;
 --photo-bg:#dde3ea;--scrim:rgba(16,24,40,.52);
 --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px rgba(16,24,40,.06);
 --r:12px;--r-sm:8px;
 --ui:'Archivo',system-ui,sans-serif;--mono:'IBM Plex Mono',Consolas,monospace}
:root[data-theme="dark"]{color-scheme:dark;
 --ground:#0e141b;--panel:#151d26;--panel-2:#1a242f;--panel-3:#223040;
 --line:#243242;--line-hi:#32445a;--ink:#e8eef5;--muted:#9aa8b8;--dim:#7a8899;
 --accent:#2dd4bf;--accent-hi:#5eead4;--accent-ink:#04211d;--accent-dim:#1f6f66;--accent-soft:rgba(45,212,191,.12);
 --live:#34d399;--live-soft:rgba(52,211,153,.10);--live-line:#1f6a52;
 --bad:#f87171;--bad-soft:rgba(248,113,113,.12);--bad-line:#7a2e2e;
 --warn:#fbbf24;--warn-soft:rgba(251,191,36,.12);--warn-line:#7a5c14;
 --photo-bg:#000;--scrim:rgba(5,9,13,.82);
 --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25)}
"""

POLISH = r"""
/* ---- shared polish (theme.py) ---- */
html{background:var(--ground)}
*{scrollbar-color:var(--line-hi) transparent}
[hidden]{display:none!important}
body{background:var(--ground);color:var(--ink)}
.kicker{font-size:10.5px;font-weight:600;letter-spacing:.14em;color:var(--dim)}
input[type=checkbox]{accent-color:var(--accent);width:16px;height:16px;flex:none}
button{border-radius:var(--r-sm);background:var(--panel);border:1px solid var(--line-hi);color:var(--ink);
 box-shadow:0 1px 1px rgba(16,24,40,.04);
 transition:background .15s,border-color .15s,color .15s,box-shadow .15s}
button:hover:not([disabled]){border-color:var(--accent);color:var(--accent)}
button.quiet{background:transparent;box-shadow:none}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);
 box-shadow:0 1px 2px rgba(15,118,110,.25)}
button.primary:hover:not([disabled]){background:var(--accent-hi);border-color:var(--accent-hi);color:var(--accent-ink)}
button.danger{color:var(--bad);border-color:var(--bad-line)}
button.danger:hover:not([disabled]){background:var(--bad);border-color:var(--bad);color:#fff}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bar{background:var(--panel);border-bottom:1px solid var(--line);box-shadow:0 4px 18px rgba(16,24,40,.04)}
.brand{display:inline-flex;align-items:center;gap:9px;font-weight:700;letter-spacing:-.01em;font-size:16px}
.brand::before{content:"";width:22px;height:22px;border-radius:7px;flex:none;
 background:radial-gradient(circle at 50% 50%,#fff 0 20%,transparent 22%),
            radial-gradient(circle at 50% 50%,transparent 0 40%,rgba(255,255,255,.55) 41% 46%,transparent 47%),var(--accent)}
aside{background:var(--panel);border-left:1px solid var(--line)}
main{background:var(--ground)}
.stage,.card,.fld,.sheet,.browser{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);
 box-shadow:var(--shadow)}
.card.look{overflow:hidden}
.stage.active{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft),var(--shadow)}
.sheet,.browser{border-color:var(--line-hi)}
.sheet-head,.bhead,.bfoot,.looks{background:var(--panel)}
.dest,.bnew{background:var(--panel-2)}
.track{background:var(--panel-3)}
.fill{background:linear-gradient(90deg,var(--accent),var(--accent-hi))}
.num .v{letter-spacing:-.02em}
.arrow .cnt{color:var(--accent)}
input[type=text],input[type=number],select{background:var(--panel);border:1px solid var(--line-hi);
 border-radius:var(--r-sm);color:var(--ink)}
.dest input,.dest .val,.bnew input,.cs input[type=number]{background:var(--panel);border-color:var(--line-hi);
 border-radius:var(--r-sm)}
.fld .val{background:var(--panel-2);border-radius:var(--r-sm)}
input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.scrim{background:var(--scrim)}
.chip{background:var(--accent-soft);border:1px solid var(--accent-dim);color:var(--accent)}
.keep{color:var(--accent);background:var(--live-soft);border:1px solid var(--live-line)}
.look-img,.shot,.lb2 img,.chip .sw{background:var(--photo-bg)}
.chip .sw{border:1px solid var(--line)}
.lb2:hover,.bitem:hover{background:var(--panel-3)}
.val.empty{text-align:left}
.lb2[aria-current=true]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.lb2[aria-current=true] .cur{color:var(--accent-ink);opacity:.75}
.toast{background:var(--ink);color:var(--panel);border-radius:var(--r-sm);box-shadow:var(--shadow)}
/* photos are judged on a dark surround whatever the theme */
.lb{background:#0b0b0b}
.lb .meta{color:#aab4c0}
.lb button{background:transparent;color:#e8eef5;border-color:#3a4654}
.lb button:hover:not([disabled]){border-color:#2dd4bf;color:#2dd4bf}
/* Light / Dark switch: one icon button, showing the theme it will switch to */
.iconbtn{width:36px;height:36px;padding:0;display:inline-grid;place-items:center;border-radius:var(--r-sm)}
.iconbtn svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.iconbtn .i-sun{display:none}
:root[data-theme="dark"] .iconbtn .i-sun{display:block}
:root[data-theme="dark"] .iconbtn .i-moon{display:none}
"""

HEAD = r"""<script>(function(){var t=null;try{t=localStorage.getItem('tr-theme');}catch(e){}
document.documentElement.setAttribute('data-theme',t==='dark'?'dark':'light');})();</script>"""

TOGGLE = r"""<button class="quiet iconbtn" id="th-btn" type="button" aria-label="Switch to dark theme" title="Switch to dark theme">
    <svg class="i-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
    <svg class="i-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>"""

JS = r"""
function setTheme(t,save){
  document.documentElement.setAttribute('data-theme',t);
  var b=document.getElementById('th-btn');
  if(b){var next=t==='dark'?'light':'dark',txt='Switch to '+next+' theme';
    b.setAttribute('aria-label',txt);b.setAttribute('title',txt);}
  if(save){try{localStorage.setItem('tr-theme',t);}catch(e){}}
}
(function(){setTheme(document.documentElement.getAttribute('data-theme')||'light',false);
  var b=document.getElementById('th-btn');
  if(b)b.onclick=function(){setTheme(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark',true);};})();
"""


def apply(page: str) -> str:
    """Fill a page's theme placeholders."""
    return (page.replace("/*@THEME_TOKENS@*/", TOKENS)
                .replace("/*@THEME_POLISH@*/", POLISH)
                .replace("<!--@THEME_HEAD@-->", HEAD)
                .replace("<!--@THEME_TOGGLE@-->", TOGGLE)
                .replace("/*@THEME_JS@*/", JS))
