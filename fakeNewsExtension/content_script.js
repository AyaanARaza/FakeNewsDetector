console.log('Content script loaded');
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.action === 'getPageText') {
    const title = document.querySelector('h1')?.innerText || document.title;
    const body  = Array.from(document.querySelectorAll('p'))
                       .map(p=>p.innerText).join(' ');
    sendResponse({ title, body });
  }
});
