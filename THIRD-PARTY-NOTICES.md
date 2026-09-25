# Third-party notices

jeva is Apache-2.0. It includes material from the project below, under the terms reproduced here.

---

## browser-use/jev-ultrafast

`jeva/snapshot.js` is derived from `jev_ultrafast/snapshot.js` in
<https://github.com/browser-use/jev-ultrafast> — the DOM snapshotter, its node-identity cache, the
`pageKey`/`guard` helpers and the viewport/visibility rules are substantially that project's work.
The surrounding driver (`jeva/browser.py`), the agent loop (`jeva/agent.py`) and the prompt
rendering are original to this repository, though they follow the same architecture.

```
MIT License

Copyright (c) 2026 Browser Use

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Note on scope: jev-ultrafast calls TypeSafe's hosted System One API for decisions and a separate
language model for text values. jeva replaces both with one local model, so no TypeSafe or
browser-harness code is used here.
