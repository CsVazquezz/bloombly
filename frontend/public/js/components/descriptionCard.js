import { state } from '../state.js';

let currentPoint = null;

export function initDescriptionCard() {
  const container = document.createElement('div');
  container.id = 'description-card';
  container.style.display = 'none';
  document.body.appendChild(container);
}

export function showDescriptionCard(point, event, coords) {
  currentPoint = point;
  const card = document.getElementById('description-card');
  
  if (!card) {
    console.error('Description card not initialized');
    return;
  }
  
  // Build the card content
  const familyColor = state.familyColors[point.Family] || '#4ED9D9';
  const genusColor = state.genusColors[point.Genus] || familyColor;
  
  card.innerHTML = `
    <div class="card-header">
      <h3>Bloom Details</h3>
      <button class="close-btn" id="closeDescriptionCard">✕</button>
    </div>
    
    <div class="card-content">
      <div class="info-section">
        <div class="info-row">
          <span class="label">Family</span>
          <div class="value-with-color">
            <div class="color-indicator" style="background-color: ${familyColor};"></div>
            <span class="value">${point.Family || 'N/A'}</span>
          </div>
        </div>
        
        <div class="info-row">
          <span class="label">Genus</span>
          <div class="value-with-color">
            <div class="color-indicator" style="background-color: ${genusColor};"></div>
            <span class="value">${point.Genus || 'N/A'}</span>
          </div>
        </div>
        
        <div class="info-row">
          <span class="label">Season</span>
          <span class="value">${point.Season || 'N/A'}</span>
        </div>
        
        <div class="info-row">
          <span class="label">Year</span>
          <span class="value">${point.year || 'N/A'}</span>
        </div>
        
        <div class="info-row">
          <span class="label">Area</span>
          <span class="value">${point.Area ? point.Area.toFixed(2) + ' km²' : 'N/A'}</span>
        </div>
      </div>
      
      <div class="info-section">
        <h4>Location</h4>
        <div class="info-row">
          <span class="label">Latitude</span>
          <span class="value">${point.lat ? point.lat.toFixed(4) + '°' : 'N/A'}</span>
        </div>
        
        <div class="info-row">
          <span class="label">Longitude</span>
          <span class="value">${point.lng ? point.lng.toFixed(4) + '°' : 'N/A'}</span>
        </div>
      </div>
      
      ${point.confidence ? `
        <div class="info-section prediction-section">
          <h4>Prediction Info</h4>
          <div class="info-row">
            <span class="label">Confidence</span>
            <span class="value confidence-value">${(point.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      ` : ''}
    </div>
  `;
  
  // Show the card with animation
  card.style.display = 'flex';
  
  // Use setTimeout to trigger animation after display is set
  setTimeout(() => {
    card.classList.add('visible');
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
