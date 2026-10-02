#!/usr/bin/env python3
"""End-to-end check in a real browser: does the canvas ever sit on a spinner?

Run:  python scripts/browser_check.py
      python scripts/browser_check.py --headed        (watch it happen)

This exists because the loading hang is invisible to pytest. It is a FRONTEND
state: any component in an event's output list is marked pending the moment the
click is dispatched, and a Sketchpad does not reliably come back out of it. The
fix is that the Sketchpads are in no event's outputs at all - `pytest` checks
that structurally, and this checks that the page actually behaves.

It fails loudly if any block is still `pending` after an interaction, or if the
server logged a traceback.
"""
from __future__ import annotations

import asyncio
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 7899
URL = f"http://127.0.0.1:{PORT}/"

SERVER = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
import app as A
A.launch(server_name="127.0.0.1", server_port={PORT},
              prevent_thread_lock=True, quiet=True)
import time
while True: time.sleep(3600)
"""


async def run(headed: bool) -> int:
    from playwright.async_api import async_playwright

    failures: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        page = await browser.new_page(viewport={"width": 1500, "height": 1200})
        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        async def pending() -> list[str]:
            return await page.evaluate(
                "() => [...document.querySelectorAll('.block')]"
                "  .filter(b => /pending|generating/.test(b.className))"
                "  .map(b => b.id)")

        async def settle(label: str, timeout_s: int = 30) -> None:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                await page.wait_for_timeout(400)
                if not await pending():
                    return
            failures.append(f"{label}: still pending after {timeout_s}s "
                            f"-> {await pending()}")

        def btn(name: str):
            """The app's own buttons. The Sketchpad toolbar has an 'Undo' too."""
            return page.locator("button:not(.icon-button)", has_text=name).first

        async def pick_draw_tool() -> None:
            # The Sketchpad starts with no tool selected and says so on itself.
            for name in ("Draw button", "Draw", "draw"):
                found = page.get_by_label(name)
                if await found.count():
                    await found.first.click()
                    return
            await page.locator("button[aria-label*='raw' i]").first.click()

        async def stroke(points) -> None:
            await page.mouse.move(*points[0])
            await page.mouse.down()
            for pt in points[1:]:
                await page.mouse.move(*pt, steps=3)
            await page.mouse.up()
            await page.wait_for_timeout(400)

        async def visible_canvas():
            return page.locator("canvas:visible").first

        await (await visible_canvas()).wait_for(state="visible", timeout=20000)
        await pick_draw_tool()
        box = await (await visible_canvas()).bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        rad = min(box["width"], box["height"]) * 0.22

        async def circle(r=rad, at=None):
            ox, oy = at or (cx, cy)
            await stroke([(ox + r * math.cos(2 * math.pi * i / 40),
                           oy + r * math.sin(2 * math.pi * i / 40))
                          for i in range(41)])

        # --- the reported sequence, several times over ----------------------- #
        await circle()
        await btn("What did I draw?").click()
        await settle("preview")

        for i in range(3):
            await btn("Add volume").click()
            await settle(f"add #{i + 1}")
            # The reported case: clicking again on a canvas nothing has touched.
            await btn("Add volume").click()
            await settle(f"add #{i + 1} on an untouched canvas")
            await circle(rad * 0.55, at=(cx - rad * 1.3 + i * rad * 0.9, cy + rad * 1.2))

        await btn("Cut volume").click()
        await settle("cut")
        await btn("Undo").click()
        await settle("undo")
        await page.screenshot(path="/tmp/browser-check-free.png", full_page=True)

        # --- half mode ------------------------------------------------------- #
        await page.get_by_role("radio", name="Half + centreline").click()
        await settle("mode switch")
        await pick_draw_tool()
        await page.wait_for_timeout(400)
        await btn("Add volume").click()
        await settle("add in half mode with nothing drawn")

        # --- click-to-place -------------------------------------------------- #
        # Back to free draw first: the four surfaces keep separate books, and a
        # rectangle clicked across the centreline is not a valid half.
        await page.get_by_role("radio", name="Free draw").click()
        await settle("back to free draw")
        await page.get_by_role("radio", name="Click the corners").click()
        await settle("style switch")
        img = page.locator("img:visible").first
        try:
            # The style swap re-renders the column, so wait for the click canvas
            # to actually be on screen before measuring it.
            await img.wait_for(state="visible", timeout=15000)
            await page.wait_for_timeout(800)
            cbox = await img.bounding_box()
        except Exception:                            # noqa: BLE001
            cbox = None
        if cbox is None:
            failures.append("click mode: the click canvas is not visible")
        else:
            w, h = cbox["width"], cbox["height"]
            for fx, fy in [(0.25, 0.8), (0.55, 0.8), (0.55, 0.35), (0.25, 0.35)]:
                # `img.click(position=...)` rather than raw mouse coordinates:
                # Playwright then scrolls it into view and waits for it to be
                # actionable, which raw coordinates do not.
                await img.click(position={"x": w * fx, "y": h * fy})
                await settle("click a corner", 15)
            note = await page.evaluate("() => document.body.innerText")
            if "corners" not in note:
                failures.append("click mode: no running note after four clicks")
            await btn("Add volume").click()
            await settle("add from clicked corners")
            await page.screenshot(path="/tmp/browser-check-clicks.png", full_page=True)
            built = await page.evaluate(
                "() => document.body.innerText.match(/clicked corners[^\\n]*/)?.[0] || ''")
            if "clicked corners" not in built:
                failures.append(f"click mode: nothing built, page said {built!r}")
            print("click mode built:", built)

        await btn("Start over").click()
        await settle("start over")

        await browser.close()

    for f in failures:
        print("FAIL:", f)
    if not failures:
        print("PASS: no component was left spinning")
    return 1 if failures else 0


def main() -> int:
    proc = subprocess.Popen([sys.executable, "-c", SERVER], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        for _ in range(60):
            time.sleep(1)
            try:
                import urllib.request
                urllib.request.urlopen(URL, timeout=2)
                break
            except Exception:                        # noqa: BLE001 - still booting
                continue
        else:
            print("FAIL: the app never came up")
            return 1
        code = asyncio.run(run("--headed" in sys.argv))
    finally:
        proc.terminate()
        try:
            log = proc.communicate(timeout=10)[0] or ""
        except subprocess.TimeoutExpired:
            proc.kill()
            log = ""
    if re.search(r"Traceback \(most recent call last\)", log):
        print("FAIL: the server logged a traceback -\n" + log[-3000:])
        code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
