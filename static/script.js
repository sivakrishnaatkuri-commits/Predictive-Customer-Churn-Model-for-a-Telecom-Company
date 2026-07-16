document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.querySelectorAll('button');
  buttons.forEach((button) => {
    button.addEventListener('mouseenter', () => button.classList.add('shadow'));
    button.addEventListener('mouseleave', () => button.classList.remove('shadow'));
  });
});
