// модуль управления спикерами, переименование, удаление и объединение
class SpeakerManager {
  constructor(appState) {
    this.state = appState;
  }

  // получение списка уникальных спикеров
  getUniqueSpeakers() {
    const segments = this.state.segments || [];
    const set = new Set();
    segments.forEach(s => {
      if (s.speaker) set.add(s.speaker);
    });
    return Array.from(set);
  }

  // переименование спикера по всей записи
  renameSpeaker(oldName, newName) {
    if (!oldName || !newName || oldName === newName) return;
    (this.state.segments || []).forEach(s => {
      if (s.speaker === oldName) {
        s.speaker = newName;
      }
    });
    this.refreshUI();
  }

  // удаление спикера и всех его реплик
  deleteSpeaker(speakerName) {
    if (!speakerName) return;
    this.state.segments = (this.state.segments || []).filter(s => s.speaker !== speakerName);
    this.refreshUI();
  }

  // слияние одного спикера с другим
  mergeSpeakers(sourceName, targetName) {
    if (!sourceName || !targetName || sourceName === targetName) return;
    (this.state.segments || []).forEach(s => {
      if (s.speaker === sourceName) {
        s.speaker = targetName;
      }
    });
    this.refreshUI();
  }

  // обновление отображения списка спикеров и текста
  refreshUI() {
    this.renderSpeakerChips();
    this.updateTranscriptElements();
  }

  // отображение панели спикеров
  renderSpeakersToolbar(segments) {
    const toolbar = document.getElementById('general-speakers-toolbar');
    if (!toolbar) return;
    toolbar.style.display = 'block';
    this.renderSpeakerChips();
  }

  // форматирование строки с подсветкой вариантов слов
  formatUncertainLine(text) {
    if (!text) return '';
    const pattern = /\[([a-zA-Zа-яА-ЯёЁ0-9\s\,\.\-]+(?:\/[a-zA-Zа-яА-ЯёЁ0-9\s\,\.\-]+)+)\]/g;
    return text.replace(pattern, (match, group) => {
      const options = group.split('/').map(o => o.trim()).filter(Boolean).slice(0, 3);
      if (options.length <= 1) return options[0] || match;

      const chosen = options[0];
      const optsHtml = options.map((opt, i) =>
        `<button type="button" class="uncertain-opt-btn ${i === 0 ? 'active' : ''}" data-val="${opt}">` +
        `<span class="opt-num">${i + 1}.</span> ${opt}${i === 0 ? ' <span class="opt-tag">реком.</span>' : ''}` +
        `</button>`
      ).join('');

      return `<span class="uncertain-word" contenteditable="false">` +
               `<span class="uncertain-text">${chosen}</span>` +
               `<span class="uncertain-tooltip" contenteditable="false">` +
                 `<span class="uncertain-tooltip-title">Варианты распознавания:</span>` +
                 optsHtml +
               `</span>` +
             `</span>`;
    });
  }

  // получение очищенного текста элемента без всплывающих подсказок
  getCleanEditableText(el) {
    const clone = el.cloneNode(true);
    clone.querySelectorAll('.uncertain-tooltip').forEach(t => t.remove());
    return clone.innerText.trim();
  }

  // отрисовка сегментов стенограммы
  renderSegments(segments, container) {
    container.innerHTML = '';
    if (!segments || segments.length === 0) {
      container.innerHTML = '<div style="color:var(--text-sub);padding:20px;text-align:center;">Речь не распознана</div>';
      return;
    }

    segments.forEach((seg, idx) => {
      const row = document.createElement('div');
      row.className = 'transcript-segment';
      row.dataset.id = seg.id || (idx + 1);

      const timeVal = seg.start_str || seg.time_str || '';
      const spk = seg.speaker || '';
      const formattedText = this.formatUncertainLine(seg.text || '');

      row.innerHTML = `
        <div class="segment-meta">
          ${timeVal ? `<span class="segment-time">[${timeVal}]</span>` : ''}
          ${spk ? `<span class="segment-speaker" data-speaker="${spk}">${spk}</span>` : ''}
        </div>
        <div class="segment-text" contenteditable="true">${formattedText}</div>
      `;

      const textEl = row.querySelector('.segment-text');
      textEl.addEventListener('input', () => {
        seg.text = this.getCleanEditableText(textEl);
      });

      container.appendChild(row);
    });
  }

  // отрисовка меток спикеров в панели управления
  renderSpeakerChips() {
    const container = document.getElementById('speakers-chips-container');
    if (!container) return;
    container.innerHTML = '';

    const speakers = this.getUniqueSpeakers();
    if (speakers.length === 0) {
      container.innerHTML = '<span style="color:var(--text-sub);font-size:12px;">Спикеры не обнаружены</span>';
      return;
    }

    speakers.forEach(spk => {
      const chip = document.createElement('div');
      chip.className = 'speaker-badge-chip';
      chip.innerHTML = `
        <span class="name-span">${spk}</span>
        <button class="chip-action-btn rename-btn" title="Переименовать" data-speaker="${spk}">
          <svg class="sci-icon" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
        </button>
        <button class="chip-action-btn merge-btn" title="Объединить" data-speaker="${spk}">
          <svg class="sci-icon" viewBox="0 0 24 24"><path d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
        </button>
        <button class="chip-action-btn delete-btn" title="Удалить реплики" data-speaker="${spk}">
          <svg class="sci-icon" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
        </button>
      `;

      chip.querySelector('.rename-btn').addEventListener('click', () => {
        window.app.openRenameModal(spk);
      });
      chip.querySelector('.merge-btn').addEventListener('click', () => {
        window.app.openMergeModal(spk);
      });
      chip.querySelector('.delete-btn').addEventListener('click', () => {
        if (confirm(`Удалить спикера "${spk}" и все связанные реплики?`)) {
          this.deleteSpeaker(spk);
        }
      });

      container.appendChild(chip);
    });
  }

  // синхронизация изменений в текстовом блоке
  updateTranscriptElements() {
    const box = document.getElementById('general-transcript-box');
    if (!box) return;

    box.querySelectorAll('.transcript-segment').forEach(segEl => {
      const segId = parseInt(segEl.dataset.id, 10);
      const segData = (this.state.segments || []).find(s => s.id === segId);
      if (!segData) {
        segEl.remove();
        return;
      }
      const spkEl = segEl.querySelector('.segment-speaker');
      if (spkEl) {
        spkEl.innerText = segData.speaker;
        spkEl.dataset.speaker = segData.speaker;
      }
    });
  }
}

window.SpeakerManager = SpeakerManager;
