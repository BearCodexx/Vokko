// модуль экспорта расшифровок и сохранения файлов
class FileExporter {
  constructor(apiBase = '/api') {
    this.apiBase = apiBase;
  }

  // универсальная точка вызова экспорта
  export(format, mode, data, filename = 'vokko_transcript') {
    return this.exportDocument(format, mode, data, filename);
  }

  // скачивание файла через серверный генератор
  async exportDocument(format, mode, data, filename = 'vokko_transcript') {
    try {
      const response = await fetch(`${this.apiBase}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          export_format: format,
          mode: mode,
          data: data,
          filename: filename
        })
      });

      if (!response.ok) {
        throw new Error('Ошибка сервера при формировании файла');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      // клиентский резервный экспорт текстовых форматов
      this._clientFallbackExport(format, mode, data, filename);
    }
  }

  // резервная выгрузка на стороне браузера
  _clientFallbackExport(format, mode, data, filename) {
    let content = '';
    let mime = 'text/plain;charset=utf-8';

    if (format === 'json') {
      const exportJson = Object.assign({}, data, {
        transcribed_by: 'Vokko',
        service: 'Транскрибировано при помощи Vokko'
      });
      content = JSON.stringify(exportJson, null, 2);
      mime = 'application/json;charset=utf-8';
    } else if (format === 'lrc' && mode === 'music') {
      const lines = [
        `[ti:${data.title || 'Композиция'}]`,
        `[ar:${data.artist || ''}]`,
        '[by:Vokko]',
        '[re:Транскрибировано при помощи Vokko]',
        ''
      ];
      (data.blocks || []).forEach(b => {
        lines.push(`// ${b.title || ''}`);
        (b.lines || []).forEach(l => {
          lines.push(`[${l.time_str || '00:00'}.00]${l.text || ''}`);
        });
        lines.push('');
      });
      content = lines.join('\n');
    } else {
      const lines = [];
      const title = data.title || (mode === 'music' ? 'Музыкальная композиция' : 'Транскрипция аудио');
      lines.push(title);
      if (mode === 'music' && data.artist) {
        lines.push(`Исполнитель, ${data.artist}`);
      }
      lines.push('Транскрибировано при помощи Vokko');
      lines.push('========================================\n');

      if (mode === 'music') {
        (data.blocks || []).forEach(b => {
          lines.push(b.title || '[Куплет]');
          (b.lines || []).forEach(l => lines.push(l.text || ''));
          lines.push('');
        });
      } else {
        (data.segments || []).forEach(s => {
          const parts = [];
          if (s.start_str) parts.push(`[${s.start_str}]`);
          if (s.speaker) parts.push(`${s.speaker},`);
          parts.push(s.text || '');
          lines.push(parts.join(' '));
        });
      }

      lines.push('\n========================================');
      lines.push('Транскрибировано при помощи Vokko');
      content = lines.join('\n');
    }

    const blob = new Blob([content], { type: mime });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }
}

window.FileExporter = FileExporter;
