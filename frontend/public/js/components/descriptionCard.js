import { state } from '../state.js';

let currentPoint = null;

export function initDescriptionCard() {
  const container = document.createElement('div');
  container.id = 'description-card';
  container.className = 'description-card';
  container.style.display = 'none';
  document.body.appendChild(container);
  console.log('[Description Card] Initialized with ID and class "description-card"');
}

export function showDescriptionCard(point, event, coords) {
  currentPoint = point;
  const card = document.getElementById('description-card');
  
  console.log('[Description Card] showDescriptionCard called', { point, card });
  
  if (!card) {
    console.error('Description card not initialized');
    return;
  }
  
  // Build the card content
  const familyColor = state.familyColors[point.Family] || '#4ED9D9';
  const genusColor = state.genusColors[point.Genus] || familyColor;
  
  // Get the genus image path
  const genusImagePath = point.Genus ? `../images/${point.Genus}.png` : null;
  
  card.innerHTML = `
    <div class="card-header">
      <h3>Bloom Details</h3>
      <button class="close-btn" id="closeDescriptionCard">✕</button>
    </div>
    
    <div class="card-content">
      ${genusImagePath ? `
        <div class="genus-image-container">
          <img src="${genusImagePath}" alt="${point.Genus}" class="genus-image" 
               onerror="this.style.display='none'; this.parentElement.style.display='none';">
        </div>
      ` : ''}
      
      <div class="taxonomy-row">
        <div class="taxonomy-item">
          <span class="label">Family</span>
          <div class="value-with-color">
            <div class="color-indicator" style="background-color: ${familyColor};"></div>
            <span class="value">${point.Family || 'N/A'}</span>
          </div>
        </div>
        
        <div class="taxonomy-item">
          <span class="label">Genus</span>
          <div class="value-with-color">
            <div class="color-indicator" style="background-color: ${genusColor};"></div>
            <span class="value">${point.Genus || 'N/A'}</span>
          </div>
        </div>
      </div>
      
      <div class="data-row">
        <div class="data-item">
          <span class="label">Area</span>
          <span class="value">${point.Area ? point.Area.toFixed(2) + ' km²' : 'N/A'}</span>
        </div>
        
        <div class="data-item">
          <span class="label">Latitude</span>
          <span class="value">${point.lat ? point.lat.toFixed(4) + '°' : 'N/A'}</span>
        </div>
        
        <div class="data-item">
          <span class="label">Longitude</span>
          <span class="value">${point.lng ? point.lng.toFixed(4) + '°' : 'N/A'}</span>
        </div>
      </div>
      
      ${point.confidence ? `
        <div class="data-item">
          <span class="label">Confidence</span>
          <span class="value confidence-value">${(point.confidence * 100).toFixed(1)}%</span>
        </div>
      ` : ''}
    </div>
  `;
  
  // Show the card with animation
  card.style.display = 'block';
  
  console.log('[Description Card] Card display set to block, adding visible class');
  
  // Use setTimeout to trigger animation after display is set
  setTimeout(() => {
    card.classList.add('visible');
    console.log('[Description Card] Visible class added, card should be shown');
  }, 10);
  
  // Attach close button event
  document.getElementById('closeDescriptionCard').addEventListener('click', hideDescriptionCard);
}

export function hideDescriptionCard() {
  const card = document.getElementById('description-card');
  if (!card) return;
  
  card.classList.remove('visible');
  
  // Wait for animation to complete before hiding
  setTimeout(() => {
    card.style.display = 'none';
    currentPoint = null;
  }, 300);
}

export function getCurrentPoint() {
  return currentPoint;
}
