// модуль эффекта печатающегося текста с контролем скорости
class TypewriterEffect {
  constructor() {
    this.speed = 18;
    this.isCancelled = false;
  }

  // печать сегментов с временной задержкой
  async playSegments(segments, container, onSegmentDone) {
    this.isCancelled = false;
    container.innerHTML = '';

    for (let i = 0; i < segments.length; i++) {
      if (this.isCancelled) {
        break;
      }
      const seg = segments[i];
      const segEl = document.createElement('div');
      segEl.className = 'transcript-segment';
      segEl.dataset.id = seg.id || (i + 1);

      let headerHtml = '<div class="segment-header">';
      if (seg.start_str || seg.time_str) {
        headerHtml += `<span class="segment-time">[${seg.start_str || seg.time_str}]</span>`;
      }
      if (seg.speaker) {
        headerHtml += `<span class="segment-speaker" data-speaker="${seg.speaker}">${seg.speaker}</span>`;
      }
      headerHtml += '</div>';

      const textEl = document.createElement('div');
      textEl.className = 'segment-text';
      textEl.contentEditable = 'true';

      segEl.innerHTML = headerHtml;
      segEl.appendChild(textEl);
      container.appendChild(segEl);

      const fullText = seg.text || '';
      for (let charIdx = 0; charIdx < fullText.length; charIdx++) {
        if (this.isCancelled) {
          textEl.textContent = fullText;
          break;
        }
        textEl.textContent += fullText[charIdx];
        await new Promise(r => setTimeout(r, this.speed));
      }

      if (onSegmentDone) {
        onSegmentDone(seg, i);
      }
    }
  }

  // прерывание текущей анимации печати
  stop() {
    this.isCancelled = true;
  }
}

window.TypewriterEffect = TypewriterEffect;
