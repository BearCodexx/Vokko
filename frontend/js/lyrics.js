// модуль отображения и редактирования музыкального текста и строф
class LyricManager {
  constructor(appState) {
    this.state = appState;
    this.initOptionHandlers();
  }

  // глобальный обработчик клика по варианту слова
  initOptionHandlers() {
    document.addEventListener('click', e => {
      const optBtn = e.target.closest('.uncertain-opt-btn');
      if (!optBtn) return;
      e.stopPropagation();
      e.preventDefault();

      const val = optBtn.dataset.val;
      const wordWrap = optBtn.closest('.uncertain-word');
      if (!wordWrap) return;

      const textSpan = wordWrap.querySelector('.uncertain-text');
      if (textSpan) textSpan.innerText = val;

      wordWrap.querySelectorAll('.uncertain-opt-btn').forEach(b => b.classList.remove('active'));
      optBtn.classList.add('active');

      const editableEl = wordWrap.closest('.lyric-line-text, .segment-text');
      if (editableEl) {
        editableEl.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
  }

  // форматирование строки текста песни
  formatUncertainLine(text) {
    if (!text) return '';
    const pattern = /\[([a-zA-Zа-яА-ЯёЁ0-9\s\,\.\-]+(?:\/[a-zA-Zа-яА-ЯёЁ0-9\s\,\.\-]+)+)\]/g;
    return text.replace(pattern, (match, group) => {
      const options = group.split('/').map(o => o.trim()).filter(Boolean);
      return options[0] || match;
    });
  }

  // извлечение чистого текста из редактируемого блока
  getCleanEditableText(el) {
    return el.innerText.trim();
  }

  // отрисовка структурных блоков песни без таймкодов
  renderBlocks(blocks, container) {
    container.innerHTML = '';
    if (!blocks || blocks.length === 0) {
      container.innerHTML = '<div style="color:var(--text-sub);padding:20px;text-align:center;">Текст песни не содержит распознанных строф</div>';
      return;
    }

    blocks.forEach((b, blockIdx) => {
      const card = document.createElement('div');
      const typeClass = (b.type || 'verse').toLowerCase();
      card.className = `lyric-block-card ${typeClass}`;
      card.dataset.blockId = b.id || (blockIdx + 1);

      const header = document.createElement('div');
      header.className = 'lyric-block-header';
      header.innerHTML = `
        <span class="lyric-badge">${b.title || 'Куплет'}</span>
        <span style="font-family:'Share Tech Mono';font-size:11px;color:var(--text-sub);">${b.lines ? b.lines.length : 0} строк</span>
      `;
      card.appendChild(header);

      const linesWrap = document.createElement('div');
      linesWrap.className = 'lyric-lines-wrapper';

      (b.lines || []).forEach((l, lineIdx) => {
        const lineEl = document.createElement('div');
        lineEl.className = 'lyric-line';

        const formattedHtml = this.formatUncertainLine(l.text || '');

        lineEl.innerHTML = `
          <div class="lyric-line-text" contenteditable="true" data-line-idx="${lineIdx}">${formattedHtml}</div>
        `;

        // синхронизация редактирования строки
        const textInput = lineEl.querySelector('.lyric-line-text');
        textInput.addEventListener('input', () => {
          l.text = this.getCleanEditableText(textInput);
        });

        linesWrap.appendChild(lineEl);
      });

      card.appendChild(linesWrap);
      container.appendChild(card);
    });
  }

  // сбор актуальных данных текста для выгрузки
  collectData() {
    return {
      title: this.state.musicTitle || 'Музыкальная композиция',
      artist: this.state.musicArtist || '',
      duration: this.state.duration || 0,
      blocks: this.state.musicBlocks || []
    };
  }
}

window.LyricManager = LyricManager;
