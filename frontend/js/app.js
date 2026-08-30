// главный управляющий модуль интерфейса Vokko
class VokkoApp {
  constructor() {
    this.currentMode = 'general';
    this.segments = [];
    this.musicBlocks = [];
    this.generalTitle = '';
    this.musicTitle = '';
    this.musicArtist = '';
    this.duration = 0;
    this.selectedFile = null;
    this.isProcessing = false;
    this.progressPollInterval = null;
    this.modelsCatalog = null;
    this.modelDownloadPolls = {};

    this.typewriter = new TypewriterEffect();
    this.speakerManager = new SpeakerManager(this);
    this.lyricManager = new LyricManager(this);
    this.exporter = new FileExporter();

    this.initTheme();
    this.bindEvents();
    this.loadModelsCatalog();
  }

  // инициализация цветовой схемы, поддержка темной, светлой и системной
  initTheme() {
    const savedTheme = localStorage.getItem('vokko_theme') || 'dark';
    this.applyTheme(savedTheme);

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
      if (localStorage.getItem('vokko_theme') === 'system') {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }
    });
  }

  // применение темы оформления
  applyTheme(theme) {
    localStorage.setItem('vokko_theme', theme);
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.theme === theme);
    });

    if (theme === 'system') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }

  // загрузка каталога доступных моделей распознавания и языковой коррекции
  async loadModelsCatalog(engine) {
    try {
      const activeEngine = engine || document.getElementById('general-engine-select')?.value || 'faster-whisper';
      const resp = await fetch(`/api/models?engine=${encodeURIComponent(activeEngine)}`);
      if (!resp.ok) return;
      this.modelsCatalog = await resp.json();
      this.populateModelSelectors();
    } catch (e) {
      console.error('Не удалось загрузить каталог моделей', e);
    }
  }

  // заполнение выпадающих списков выбора моделей
  populateModelSelectors() {
    if (!this.modelsCatalog) return;

    const asrSelects = [
      document.getElementById('general-asr-select'),
      document.getElementById('music-asr-select')
    ];

    const llmSelects = [
      document.getElementById('general-llm-select'),
      document.getElementById('music-llm-select')
    ];

    const activeAsr = this.modelsCatalog.active_asr || 'large-v3';
    const activeLlm = this.modelsCatalog.active_llm || 'none';

    asrSelects.forEach(sel => {
      if (!sel) return;
      sel.innerHTML = '';
      (this.modelsCatalog.asr_models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        const status = m.downloaded ? '✓' : `[Скачать ${m.ssd}]`;
        opt.textContent = `${m.name} (${m.speed}) ${status}`;
        if (m.id === activeAsr) opt.selected = true;
        sel.appendChild(opt);
      });
    });

    llmSelects.forEach(sel => {
      if (!sel) return;
      sel.innerHTML = '';
      (this.modelsCatalog.llm_models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        let tag = '';
        if (m.id === 'none') {
          tag = '[Без LLM]';
        } else if (m.id === 'openrouter') {
          tag = '[Облачный API]';
        } else if (m.id === 'custom-ollama') {
          tag = '[Локальный сервер]';
        } else {
          tag = m.downloaded ? '✓' : `[Скачать ${m.ssd}]`;
        }
        opt.textContent = `${m.name} ${tag}`;
        if (m.id === activeLlm) opt.selected = true;
        sel.appendChild(opt);
      });
    });

    this.updateLlmAdvancedPanel('general');
    this.updateLlmAdvancedPanel('music');
  }

  // переключение видимости расширенных полей параметров LLM
  updateLlmAdvancedPanel(mode) {
    const prefix = mode === 'music' ? 'music' : 'general';
    const sel = document.getElementById(`${prefix}-llm-select`);
    if (!sel) return;
    const val = sel.value;

    const advPanel = document.getElementById(`${prefix}-llm-advanced-panel`);
    const orBox = document.getElementById(`${prefix}-openrouter-box`);
    const olBox = document.getElementById(`${prefix}-ollama-box`);
    const layBox = document.getElementById(`${prefix}-layers-box`);

    if (!advPanel) return;

    if (val === 'none') {
      advPanel.style.display = 'none';
      return;
    }

    advPanel.style.display = 'block';
    if (layBox) layBox.style.display = 'block';

    const batchBox = document.getElementById('general-batch-box');
    if (batchBox) {
      batchBox.style.display = prefix === 'general' ? 'block' : 'none';
    }

    if (val === 'openrouter') {
      if (orBox) orBox.style.display = 'block';
      if (olBox) olBox.style.display = 'none';
      const keyInp = document.getElementById(`${prefix}-openrouter-key`);
      const modelInp = document.getElementById(`${prefix}-openrouter-model`);
      if (keyInp && this.modelsCatalog?.openrouter_api_key && !keyInp.value) {
        keyInp.value = this.modelsCatalog.openrouter_api_key;
      }
      if (modelInp && this.modelsCatalog?.openrouter_model && !modelInp.value) {
        modelInp.value = this.modelsCatalog.openrouter_model;
      }
    } else if (val === 'custom-ollama') {
      if (orBox) orBox.style.display = 'none';
      if (olBox) olBox.style.display = 'block';
      this.fetchAndPopulateOllamaModels(prefix);
    } else {
      if (orBox) orBox.style.display = 'none';
      if (olBox) olBox.style.display = 'none';
    }
  }

  // загрузка списка моделей из локального сервера ollama
  async fetchAndPopulateOllamaModels(prefix) {
    try {
      const resp = await fetch('/api/ollama/models');
      if (!resp.ok) return;
      const data = await resp.json();
      const sel = document.getElementById(`${prefix}-ollama-model-select`);
      if (!sel) return;
      sel.innerHTML = '';
      const models = data.models || [];
      if (models.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Модели в Ollama не найдены';
        sel.appendChild(opt);
        return;
      }
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === data.active || m === this.modelsCatalog?.ollama_model) opt.selected = true;
        sel.appendChild(opt);
      });
    } catch (e) {}
  }

  // открытие модального окна каталога и докачки моделей
  openModelsModal() {
    const modal = document.getElementById('models-modal');
    if (!modal) return;
    this.renderModelsModalContent();
    modal.classList.add('active');
  }

  // закрытие окна моделей
  closeModelsModal() {
    const modal = document.getElementById('models-modal');
    if (modal) modal.classList.remove('active');
  }

  // отрисовка содержимого карточек моделей в модальном окне
  renderModelsModalContent() {
    const body = document.getElementById('models-modal-body');
    if (!body || !this.modelsCatalog) return;

    body.innerHTML = `
      <div>
        <div class="models-section-title">
          <svg class="sci-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          МОДЕЛИ РАСПОЗНАВАНИЯ РЕЧИ (WHISPER ASR)
        </div>
        <div class="models-grid-list" id="modal-asr-grid"></div>
      </div>
      <div>
        <div class="models-section-title">
          <svg class="sci-icon" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
          МОДЕЛИ НЕЙРОКОРРЕКЦИИ (LLM / SLM GGUF)
        </div>
        <div class="models-grid-list" id="modal-llm-grid"></div>
      </div>
    `;

    const asrGrid = document.getElementById('modal-asr-grid');
    (this.modelsCatalog.asr_models || []).forEach(m => {
      const card = this.createModelCard(m);
      asrGrid.appendChild(card);
    });

    const llmGrid = document.getElementById('modal-llm-grid');
    (this.modelsCatalog.llm_models || []).forEach(m => {
      const card = this.createModelCard(m);
      llmGrid.appendChild(card);
    });
  }

  // создание одной карточки модели для менеджера
  createModelCard(m) {
    const el = document.createElement('div');
    el.className = 'model-card-item';
    el.id = `model-card-${m.id}`;

    const isReady = m.downloaded || m.id === 'none' || m.id === 'custom-ollama' || m.id === 'openrouter';

    el.innerHTML = `
      <div class="model-card-header">
        <span class="model-card-name">${m.name}</span>
        <span class="model-badge">${m.category || m.type}</span>
      </div>
      <div class="model-card-specs">
        <span>Качество, ${m.quality}</span>
        <span>Скорость, ${m.speed}</span>
        <span>Память, ${m.vram} VRAM | ${m.ssd} SSD</span>
      </div>
      <div class="model-card-footer" id="model-footer-${m.id}">
        ${isReady ? `
          <span class="model-status-ready">
            <svg class="sci-icon" viewBox="0 0 24 24" style="width:14px;height:14px;"><polyline points="20 6 9 17 4 12"></polyline></svg>
            Готова к работе
          </span>
        ` : `
          <button class="btn-download-model" onclick="window.app.triggerModelDownload('${m.id}')">
            Скачать (${m.ssd})
          </button>
        `}
      </div>
    `;
    return el;
  }

  // запуск скачивания выбранной модели
  async triggerModelDownload(modelId) {
    const footer = document.getElementById(`model-footer-${modelId}`);
    if (footer) {
      footer.innerHTML = `
        <span style="font-family:'Share Tech Mono';font-size:11px;color:var(--neon-green);" id="dl-label-${modelId}">
          Подготовка...
        </span>
      `;
    }

    const currentEngine = document.getElementById('general-engine-select')?.value || 'faster-whisper';

    try {
      await fetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, engine: currentEngine })
      });
      this.pollModelDownload(modelId);
    } catch (e) {
      if (footer) footer.innerHTML = `<span style="color:#f87171;font-size:11px;">Сбой запуска</span>`;
    }
  }

  // периодический опрос прогресса скачивания модели
  pollModelDownload(modelId) {
    if (this.modelDownloadPolls[modelId]) {
      clearInterval(this.modelDownloadPolls[modelId]);
    }

    this.modelDownloadPolls[modelId] = setInterval(async () => {
      try {
        const resp = await fetch(`/api/models/progress/${modelId}`);
        if (!resp.ok) return;
        const prog = await resp.json();
        const label = document.getElementById(`dl-label-${modelId}`);

        if (prog.status === 'downloading') {
          if (label) label.innerText = `Загрузка: ${prog.percent.toFixed(1)}%`;
        } else if (prog.status === 'completed') {
          clearInterval(this.modelDownloadPolls[modelId]);
          delete this.modelDownloadPolls[modelId];
          const footer = document.getElementById(`model-footer-${modelId}`);
          if (footer) {
            footer.innerHTML = `
              <span class="model-status-ready">
                <svg class="sci-icon" viewBox="0 0 24 24" style="width:14px;height:14px;"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Готова к работе
              </span>
            `;
          }
          await this.loadModelsCatalog();
        } else if (prog.status === 'error') {
          clearInterval(this.modelDownloadPolls[modelId]);
          delete this.modelDownloadPolls[modelId];
          const footer = document.getElementById(`model-footer-${modelId}`);
          if (footer) footer.innerHTML = `<span style="color:#f87171;font-size:11px;">Ошибка скачивания</span>`;
        }
      } catch (e) {
        clearInterval(this.modelDownloadPolls[modelId]);
      }
    }, 800);
  }

  // привязка обработчиков интерфейса
  bindEvents() {
    // переключение тем оформления
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.addEventListener('click', () => this.applyTheme(btn.dataset.theme));
    });

    // выбор режима транскрибации
    document.getElementById('card-mode-general').addEventListener('click', () => this.switchMode('general'));
    document.getElementById('card-mode-music').addEventListener('click', () => this.switchMode('music'));

    // настройка зоны перетаскивания файлов
    this.initDropZone('general-drop-zone', 'general-file-input', file => {
      this.selectedFile = file;
      document.getElementById('general-file-name').innerText = file.name;
      this.hideError('general');
    });

    this.initDropZone('music-drop-zone', 'music-file-input', file => {
      this.selectedFile = file;
      document.getElementById('music-file-name').innerText = file.name;
      this.hideError('music');
    });

    // закрытие сообщений об ошибках
    document.getElementById('general-error-dismiss').addEventListener('click', () => this.hideError('general'));
    document.getElementById('music-error-dismiss').addEventListener('click', () => this.hideError('music'));

    // управление модальным окном моделей
    document.querySelectorAll('.btn-models-manager').forEach(btn => {
      btn.addEventListener('click', () => this.openModelsModal());
    });

    const closeBtn = document.getElementById('models-modal-close');
    if (closeBtn) closeBtn.addEventListener('click', () => this.closeModelsModal());

    const doneBtn = document.getElementById('models-modal-done');
    if (doneBtn) doneBtn.addEventListener('click', () => this.closeModelsModal());

    // привязка выпадающих списков LLM и слоев температуры
    ['general', 'music'].forEach(prefix => {
      const slider = document.getElementById(`${prefix}-llm-layers`);
      const badge = document.getElementById(`${prefix}-layers-badge`);
      const llmSel = document.getElementById(`${prefix}-llm-select`);
      if (slider && badge) {
        const layerDescriptions = {
          '1': '1 (Оптимальный, 1 проход)',
          '2': '2 (2 слоя: стабильный + умеренный)',
          '3': '3 (3 слоя: строгий + средний + творческий)',
          '4': '4 (4 слоя: глубокий синтез)',
          '5': '5 (5 слоев: максимальный консенсус)'
        };
        slider.addEventListener('input', e => {
          badge.textContent = layerDescriptions[e.target.value] || `${e.target.value} слоев`;
        });
      }
      if (llmSel) {
        llmSel.addEventListener('change', () => {
          this.updateLlmAdvancedPanel(prefix);
        });
      }
    });

    // привязка слайдера размера батча
    const batchSlider = document.getElementById('general-batch-size');
    const batchBadge = document.getElementById('general-batch-badge');
    const batchVal = document.getElementById('general-batch-val');
    if (batchSlider) {
      batchSlider.addEventListener('input', e => {
        const v = e.target.value;
        if (batchBadge) batchBadge.textContent = `${v} мин (до завершения реплики)`;
        if (batchVal) batchVal.textContent = `${v} мин`;
      });
    }

    // привязка слайдера поиска по лучам beam size
    const beamSlider = document.getElementById('general-beam-size');
    const beamBadge = document.getElementById('general-beam-badge');
    const beamVal = document.getElementById('general-beam-val');
    if (beamSlider) {
      const beamDesc = {
        '1': '1 (Сверхбыстрый)',
        '2': '2 (Быстрый)',
        '3': '3 (Оптимальный)',
        '4': '4 (Тщательный)',
        '5': '5 (Глубокий поиск)'
      };
      beamSlider.addEventListener('input', e => {
        const v = e.target.value;
        if (beamBadge) beamBadge.textContent = beamDesc[v] || `${v}`;
        if (beamVal) beamVal.textContent = `${v}`;
      });
    }

    // смена движка Whisper и обновление доступности моделей
    const engineSel = document.getElementById('general-engine-select');
    if (engineSel) {
      engineSel.addEventListener('change', () => {
        this.loadModelsCatalog(engineSel.value);
      });
    }

    // запуск обработки
    document.getElementById('btn-start-general').addEventListener('click', () => this.startGeneralTranscription());
    document.getElementById('btn-start-music').addEventListener('click', () => this.startMusicTranscription());

    // управление выпадающим списком экспорта
    document.querySelectorAll('.export-trigger-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const menu = btn.nextElementSibling;
        menu.classList.toggle('show');
      });
    });

    document.addEventListener('click', () => {
      document.querySelectorAll('.export-menu').forEach(m => m.classList.remove('show'));
    });

    // обработка кнопок экспорта
    document.querySelectorAll('.export-menu-item').forEach(item => {
      item.addEventListener('click', e => {
        const fmt = e.currentTarget.dataset.format;
        const mode = e.currentTarget.dataset.mode || this.currentMode;
        this.handleExport(fmt, mode);
      });
    });

    this.initModalActions();
  }

  // переключение режима работы
  switchMode(mode) {
    this.currentMode = mode;
    this.selectedFile = null;

    document.getElementById('card-mode-general').classList.toggle('active', mode === 'general');
    document.getElementById('card-mode-music').classList.toggle('active', mode === 'music');

    document.getElementById('workspace-general').style.display = mode === 'general' ? 'block' : 'none';
    document.getElementById('workspace-music').style.display = mode === 'music' ? 'block' : 'none';

    document.getElementById('general-file-name').innerText = 'Перетащите медиафайл или нажмите для выбора';
    document.getElementById('music-file-name').innerText = 'Перетащите аудиофайл песни (MP3, WAV, FLAC, M4A, AAC, OGG)';
    document.getElementById('general-url-input').value = '';

    this.hideError('general');
    this.hideError('music');
  }

  // настройка зоны перетаскивания файлов
  initDropZone(dropZoneId, inputId, onFileSelected) {
    const zone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);

    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());

    input.addEventListener('change', e => {
      if (e.target.files && e.target.files[0]) {
        onFileSelected(e.target.files[0]);
      }
    });

    zone.addEventListener('dragover', e => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(evt => {
      zone.addEventListener(evt, () => zone.classList.remove('drag-over'));
    });

    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        input.files = e.dataTransfer.files;
        onFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // сохранение настроек модели если указаны кастомные параметры
  async syncCustomLlmSettings(mode) {
    const prefix = mode === 'music' ? 'music' : 'general';
    const llmModel = document.getElementById(`${prefix}-llm-select`)?.value || 'none';
    const payload = { active_llm: llmModel };

    if (llmModel === 'openrouter') {
      const key = document.getElementById(`${prefix}-openrouter-key`)?.value.trim();
      const model = document.getElementById(`${prefix}-openrouter-model`)?.value.trim();
      if (key) payload.openrouter_api_key = key;
      if (model) payload.openrouter_model = model;
    } else if (llmModel === 'custom-ollama') {
      const oModel = document.getElementById(`${prefix}-ollama-model-select`)?.value;
      if (oModel) payload.ollama_model = oModel;
    }

    try {
      await fetch('/api/models/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } catch (e) {}
  }

  // запуск транскрибации видео и звуковых файлов
  async startGeneralTranscription() {
    if (this.isProcessing) return;
    const urlInput = document.getElementById('general-url-input').value.trim();
    if (!this.selectedFile && !urlInput) {
      this.showError('general', 'Пожалуйста, выберите файл или вставьте ссылку на видео');
      return;
    }

    this.hideError('general');
    const timecodes = document.getElementById('toggle-timecodes').checked;
    const diarization = document.getElementById('toggle-diarization').checked;
    const asrModel = document.getElementById('general-asr-select')?.value || 'large-v3';
    const llmModel = document.getElementById('general-llm-select')?.value || 'none';
    const llmLayers = parseInt(document.getElementById('general-llm-layers')?.value || '1', 10);
    const batchMinutes = parseFloat(document.getElementById('general-batch-size')?.value || '8');
    const engineType = document.getElementById('general-engine-select')?.value || 'faster-whisper';
    const beamSize = parseInt(document.getElementById('general-beam-size')?.value || '1', 10);
    const taskId = this.generateTaskId();

    await this.syncCustomLlmSettings('general');

    this.showGeneralProgress();
    this.startProgressPolling(taskId, 'general');
    this.isProcessing = true;

    try {
      const formData = new FormData();
      if (this.selectedFile) {
        formData.append('file', this.selectedFile);
      } else {
        formData.append('url', urlInput);
      }
      formData.append('task_id', taskId);
      formData.append('enable_timecodes', timecodes);
      formData.append('enable_diarization', diarization);
      formData.append('asr_model', asrModel);
      formData.append('llm_model', llmModel);
      formData.append('llm_layers', llmLayers);
      formData.append('batch_minutes', batchMinutes);
      formData.append('whisper_engine_type', engineType);
      formData.append('beam_size', beamSize);

      const response = await fetch('/api/transcribe/general', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Ошибка при обработке дорожки');
      }

      const result = await response.json();
      this.segments = result.segments || [];
      this.duration = result.duration || 0;
      this.generalTitle = result.title || (this.selectedFile ? this.selectedFile.name.replace(/\.[^/.]+$/, '') : 'audio');

      this.renderGeneralResults(result, diarization);
    } catch (err) {
      this.showError('general', err.message || 'Сбой при транскрибации');
    } finally {
      this.stopProgressPolling();
      this.hideGeneralProgress();
      this.isProcessing = false;
    }
  }

  // запуск транскрибации музыки с разделением дорожек Demucs
  async startMusicTranscription() {
    if (this.isProcessing) return;
    if (!this.selectedFile) {
      this.showError('music', 'Пожалуйста, перетащите или выберите аудиофайл песни (MP3, WAV, FLAC, M4A)');
      return;
    }

    this.hideError('music');
    const asrModel = document.getElementById('music-asr-select')?.value || 'large-v3';
    const llmModel = document.getElementById('music-llm-select')?.value || 'none';
    const llmLayers = parseInt(document.getElementById('music-llm-layers')?.value || '1', 10);
    const taskId = this.generateTaskId();

    await this.syncCustomLlmSettings('music');

    this.showMusicProgress();
    this.startProgressPolling(taskId, 'music');
    this.isProcessing = true;

    try {
      const formData = new FormData();
      formData.append('file', this.selectedFile);
      formData.append('task_id', taskId);
      formData.append('asr_model', asrModel);
      formData.append('llm_model', llmModel);
      formData.append('llm_layers', llmLayers);

      const response = await fetch('/api/transcribe/music', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Ошибка при извлечении вокала');
      }

      const result = await response.json();
      this.musicBlocks = result.blocks || [];
      this.musicTitle = result.title || (this.selectedFile ? this.selectedFile.name.replace(/\.[^/.]+$/, '') : 'song');
      this.musicArtist = result.artist || '';
      this.duration = result.duration || 0;

      this.renderMusicResults(result);
    } catch (err) {
      this.showError('music', err.message || 'Сбой при обработке музыки');
    } finally {
      this.stopProgressPolling();
      this.hideMusicProgress();
      this.isProcessing = false;
    }
  }

  // периодический опрос этапов выполнения
  startProgressPolling(taskId, mode) {
    this.stopProgressPolling();
    this.progressPollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/progress/${taskId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (mode === 'general') {
          this.updateGeneralStep(data.stage, data.total_stages);
        } else {
          this.updateMusicStep(data.stage, data.total_stages);
        }
      } catch (e) {}
    }, 500);
  }

  stopProgressPolling() {
    if (this.progressPollInterval) {
      clearInterval(this.progressPollInterval);
      this.progressPollInterval = null;
    }
  }

  updateGeneralStep(stage, total) {
    const steps = [
      document.getElementById('gen-step-1'),
      document.getElementById('gen-step-2'),
      document.getElementById('gen-step-3'),
      document.getElementById('gen-step-4')
    ];
    const fill = document.getElementById('general-progress-fill');

    steps.forEach((el, idx) => {
      if (!el) return;
      if (idx + 1 < stage) {
        el.className = 'step-item completed';
      } else if (idx + 1 === stage) {
        el.className = 'step-item active';
      } else {
        el.className = 'step-item';
      }
    });

    if (fill) {
      const pct = Math.min(100, Math.round((stage / (total || 4)) * 100));
      fill.style.width = `${pct}%`;
    }
  }

  updateMusicStep(stage, total) {
    const steps = [
      document.getElementById('mus-step-1'),
      document.getElementById('mus-step-2'),
      document.getElementById('mus-step-3'),
      document.getElementById('mus-step-4')
    ];
    const fill = document.getElementById('music-progress-fill');

    steps.forEach((el, idx) => {
      if (!el) return;
      if (idx + 1 < stage) {
        el.className = 'step-item completed';
      } else if (idx + 1 === stage) {
        el.className = 'step-item active';
      } else {
        el.className = 'step-item';
      }
    });

    if (fill) {
      const pct = Math.min(100, Math.round((stage / (total || 4)) * 100));
      fill.style.width = `${pct}%`;
    }
  }

  showGeneralProgress() {
    document.getElementById('general-input-form').style.display = 'none';
    document.getElementById('general-processing-hud').style.display = 'block';
    document.getElementById('general-result-panel').style.display = 'none';
    this.updateGeneralStep(1, 4);
  }

  hideGeneralProgress() {
    document.getElementById('general-processing-hud').style.display = 'none';
    document.getElementById('general-input-form').style.display = 'block';
  }

  showMusicProgress() {
    document.getElementById('music-input-form').style.display = 'none';
    document.getElementById('music-processing-hud').style.display = 'block';
    document.getElementById('music-result-panel').style.display = 'none';
    this.updateMusicStep(1, 4);
  }

  hideMusicProgress() {
    document.getElementById('music-processing-hud').style.display = 'none';
    document.getElementById('music-input-form').style.display = 'block';
  }

  renderGeneralResults(data, hasDiarization) {
    document.getElementById('general-result-panel').style.display = 'block';
    const container = document.getElementById('general-transcript-box');
    container.innerHTML = '';

    if (hasDiarization) {
      this.speakerManager.renderSpeakersToolbar(this.segments);
    } else {
      document.getElementById('general-speakers-toolbar').style.display = 'none';
    }

    this.speakerManager.renderSegments(this.segments, container);
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  renderMusicResults(data) {
    document.getElementById('music-result-panel').style.display = 'block';
    const container = document.getElementById('music-lyrics-container');
    this.lyricManager.renderBlocks(this.musicBlocks, container);
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  showError(mode, message) {
    const box = document.getElementById(`${mode}-error-box`);
    const msg = document.getElementById(`${mode}-error-message`);
    if (box && msg) {
      if (message && (message.includes('18+') || message.includes('cookies.txt') || message.includes('Get cookies.txt') || message.includes('возрастное ограничение'))) {
        box.classList.add('with-guide');
        msg.innerHTML = `
          <div class="cookies-guide-card">
            <div class="guide-title">ВИДЕО ПОМЕЧЕНО КАК 18+ (ВОЗРАСТНОЕ ОГРАНИЧЕНИЕ)</div>
            <div>YouTube блокирует прямое скачивание этого видео без подтверждения возраста. Чтобы скачивать 18+ видео напрямую по ссылке:</div>
            <ol>
              <li>Установите расширение <b>«Get cookies.txt LOCALLY»</b> в ваш браузер (Chrome / Zen / Firefox / Edge).</li>
              <li>Откройте сайт <b>YouTube</b> в браузере и нажмите на значок расширения.</li>
              <li>Экспортируйте файл, назовите его <code>cookies.txt</code>.</li>
              <li>Положите файл <code>cookies.txt</code> в папку проекта <b>Vokko</b>.</li>
            </ol>
            <div class="guide-alt">Либо просто скачайте аудио/видео файл в браузере и перетащите его мышкой в окно загрузки Vokko.</div>
          </div>
        `;
      } else {
        box.classList.remove('with-guide');
        msg.innerText = message;
      }
      box.style.display = 'flex';
    }
  }

  hideError(mode) {
    const box = document.getElementById(`${mode}-error-box`);
    if (box) {
      box.classList.remove('with-guide');
      box.style.display = 'none';
    }
  }

  cleanFileName(title) {
    if (!title) return 'track';
    return title.replace(/[/\\?%*:|"<>]/g, '_').trim();
  }

  handleExport(format, mode) {
    if (mode === 'general') {
      const data = {
        title: this.generalTitle,
        duration: this.duration,
        segments: this.segments
      };
      const cleanTitle = this.cleanFileName(this.generalTitle);
      const filename = `vokko_transcribe_${cleanTitle}`;
      this.exporter.export(format, mode, data, filename);
    } else {
      const data = this.lyricManager.collectData();
      const cleanTitle = this.cleanFileName(this.musicTitle);
      const filename = `vokko_lyrics_${cleanTitle}`;
      this.exporter.export(format, mode, data, filename);
    }
  }

  generateTaskId() {
    return 'task_' + Math.random().toString(36).substring(2, 9);
  }

  initModalActions() {
    const modal = document.getElementById('vokko-modal');
    const cancelBtn = document.getElementById('modal-cancel-btn');
    const confirmBtn = document.getElementById('modal-confirm-btn');

    cancelBtn.addEventListener('click', () => {
      modal.classList.remove('active');
    });

    confirmBtn.addEventListener('click', () => {
      if (this.currentModalCallback) {
        this.currentModalCallback();
      }
      modal.classList.remove('active');
    });
  }

  openRenameModal(speakerName) {
    const modal = document.getElementById('vokko-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');

    title.innerText = `Переименовать спикера: ${speakerName}`;
    body.innerHTML = `
      <input type="text" id="rename-input-val" class="sci-input" style="width:100%;" value="${speakerName}" />
    `;

    this.currentModalCallback = () => {
      const val = document.getElementById('rename-input-val').value.trim();
      if (val) {
        this.speakerManager.renameSpeaker(speakerName, val);
      }
    };

    modal.classList.add('active');
    setTimeout(() => document.getElementById('rename-input-val').focus(), 50);
  }

  openMergeModal(sourceSpeaker) {
    const modal = document.getElementById('vokko-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');

    const otherSpeakers = this.speakerManager.getUniqueSpeakers().filter(s => s !== sourceSpeaker);
    if (otherSpeakers.length === 0) {
      this.showError('general', 'Нет других спикеров для объединения');
      return;
    }

    title.innerText = `Объединить спикера "${sourceSpeaker}"`;
    let selectOptions = otherSpeakers.map(s => `<option value="${s}">${s}</option>`).join('');

    body.innerHTML = `
      <p style="margin-bottom:10px;font-size:13px;color:var(--text-muted);">Перенести все реплики спикера <b>${sourceSpeaker}</b> к:</p>
      <select id="merge-select-val" class="sci-input" style="width:100%;">${selectOptions}</select>
    `;

    this.currentModalCallback = () => {
      const target = document.getElementById('merge-select-val').value;
      if (target) {
        this.speakerManager.mergeSpeakers(sourceSpeaker, target);
      }
    };

    modal.classList.add('active');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new VokkoApp();
});
