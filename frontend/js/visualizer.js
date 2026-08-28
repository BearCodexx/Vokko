// холст анимации звуковых спектров и гармоник
function initAudioVisualizer() {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight * 0.5;
  }
  window.addEventListener('resize', resize);
  resize();

  let step = 0;

  function renderFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    step += 0.035;

    const baseHeight = canvas.height * 0.75;
    const barWidth = 4;
    const barGap = 6;
    const totalStep = barWidth + barGap;
    const count = Math.floor(canvas.width / totalStep);

    // отрисовка столбиков эквалайзера частот
    for (let i = 0; i < count; i++) {
      const x = i * totalStep;
      const dist = Math.abs(canvas.width / 2 - x) / (canvas.width / 2);
      const maxHeight = (1 - dist * 0.7) * 70;
      const dynamicHeight = Math.sin(step * 2.2 + i * 0.35) * maxHeight * (0.3 + Math.random() * 0.7);

      ctx.fillStyle = 'rgba(0, 255, 170, 0.4)';
      ctx.fillRect(x, baseHeight - dynamicHeight / 2, barWidth, Math.max(2, dynamicHeight));
    }

    // отрисовка волновых нитей сигнала
    for (let j = 0; j < 3; j++) {
      ctx.beginPath();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = `rgba(0, 255, 170, ${0.45 - j * 0.12})`;

      for (let x = 0; x < canvas.width; x += 14) {
        const y = baseHeight - 40 + Math.sin(x * 0.005 + step + j * 0.8) * 40 + Math.sin(x * 0.015 - step * 0.7) * 18;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    requestAnimationFrame(renderFrame);
  }

  renderFrame();
}

window.addEventListener('DOMContentLoaded', initAudioVisualizer);
