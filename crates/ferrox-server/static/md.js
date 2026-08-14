// A small markdown renderer that builds DOM nodes directly.
//
// It never produces a string of HTML, so there is no step at which
// model output could be parsed as markup. Every literal character the
// model emitted — `<script>` included — arrives on screen through
// `createTextNode` and stays text. That property is the reason this
// exists at all rather than a library: a renderer that emits HTML would
// need a sanitiser behind it, and the sanitiser would be the thing to
// get wrong.
//
// Supported: fenced code blocks (with a copy button), ATX headings,
// bullet and ordered lists, blockquotes, horizontal rules, paragraphs,
// and inline code / bold / italic / links. Everything else renders as
// the literal text the model wrote, which is the correct failure mode.

import { el, copyButton } from './dom.js';

const FENCE = /^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;
const QUOTE = /^\s*>\s?(.*)$/;
const RULE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;

/** @returns {DocumentFragment} */
export function renderMarkdown(source) {
  const frag = document.createDocumentFragment();
  const lines = String(source ?? '').split('\n');
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    const fence = FENCE.exec(line);
    if (fence) {
      const [, marker, lang] = fence;
      const body = [];
      i += 1;
      // An unterminated fence is the normal state mid-stream: the model
      // has opened a block and has not closed it yet. Rendering to the
      // end of the buffer shows the code as it arrives instead of
      // hiding it until the closing fence lands.
      while (i < lines.length && !new RegExp(`^\\s*${marker[0]}{${marker.length},}\\s*$`).test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1; // the closing fence, if there was one
      frag.appendChild(codeBlock(body.join('\n'), lang));
      continue;
    }

    if (!line.trim()) { i += 1; continue; }

    if (RULE.test(line)) {
      frag.appendChild(el('hr'));
      i += 1;
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      frag.appendChild(inlineInto(el(`h${heading[1].length}`), heading[2]));
      i += 1;
      continue;
    }

    if (BULLET.test(line) || ORDERED.test(line)) {
      const ordered = !BULLET.test(line);
      const list = el(ordered ? 'ol' : 'ul');
      while (i < lines.length) {
        const match = ordered ? ORDERED.exec(lines[i]) : BULLET.exec(lines[i]);
        if (!match) break;
        list.appendChild(inlineInto(el('li'), ordered ? match[2] : match[1]));
        i += 1;
      }
      frag.appendChild(list);
      continue;
    }

    if (QUOTE.test(line)) {
      const quoted = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quoted.push(QUOTE.exec(lines[i])[1]);
        i += 1;
      }
      const quote = el('blockquote');
      quote.appendChild(renderMarkdown(quoted.join('\n')));
      frag.appendChild(quote);
      continue;
    }

    // Paragraph: everything up to a blank line or the start of another
    // block. Soft line breaks inside it become <br>, which is what a
    // chat transcript wants even though CommonMark would join them.
    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      paragraph.push(lines[i]);
      i += 1;
    }
    const p = el('p');
    paragraph.forEach((text, index) => {
      if (index > 0) p.appendChild(el('br'));
      inlineInto(p, text);
    });
    frag.appendChild(p);
  }

  return frag;
}

function isBlockStart(line) {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    BULLET.test(line) ||
    ORDERED.test(line) ||
    QUOTE.test(line) ||
    RULE.test(line)
  );
}

function codeBlock(code, lang) {
  const pre = el('pre', {}, [el('code', { text: code })]);
  const head = el('div', { class: 'codeblock-head' }, [
    el('span', { text: lang || 'text' }),
    el('span', { class: 'spacer' }),
    copyButton('copy', () => code),
  ]);
  return el('div', { class: 'codeblock' }, [head, pre]);
}

// Inline spans, matched in one pass. Ordering matters: the code-span
// alternative comes first so backticks win over emphasis inside them.
const INLINE = new RegExp(
  [
    '(`+)([\\s\\S]*?)\\1', // 1,2  code span
    '\\*\\*([\\s\\S]+?)\\*\\*', // 3   bold
    '__([\\s\\S]+?)__', // 4   bold
    '\\*([^*\\n]+?)\\*', // 5   italic
    '_([^_\\n]+?)_', // 6   italic
    '\\[([^\\]]*)\\]\\(([^()\\s]+)\\)', // 7,8 link
  ].join('|'),
  'g',
);

function inlineInto(parent, text) {
  const src = String(text ?? '');
  INLINE.lastIndex = 0;
  let cursor = 0;
  let match;
  while ((match = INLINE.exec(src)) !== null) {
    if (match.index > cursor) {
      parent.appendChild(document.createTextNode(src.slice(cursor, match.index)));
    }
    if (match[2] !== undefined) {
      parent.appendChild(el('code', { text: match[2] }));
    } else if (match[3] !== undefined || match[4] !== undefined) {
      parent.appendChild(el('strong', { text: match[3] ?? match[4] }));
    } else if (match[5] !== undefined || match[6] !== undefined) {
      parent.appendChild(el('em', { text: match[5] ?? match[6] }));
    } else if (match[8] !== undefined) {
      parent.appendChild(link(match[7], match[8]));
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < src.length) {
    parent.appendChild(document.createTextNode(src.slice(cursor)));
  }
  return parent;
}

/**
 * A link, but only to a scheme that cannot execute.
 *
 * `javascript:` and `data:` URLs in model output are the classic way a
 * "harmless" markdown renderer becomes an execution sink, so anything
 * that is not plainly http(s) or same-document renders as the literal
 * text the model wrote.
 */
function link(label, href) {
  const safe = /^(https?:\/\/|mailto:|[/#])/i.test(href);
  if (!safe) return document.createTextNode(`[${label}](${href})`);
  return el('a', {
    href,
    target: '_blank',
    rel: 'noopener noreferrer nofollow',
    text: label || href,
  });
}
