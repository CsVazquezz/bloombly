const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve data files BEFORE static middleware
app.use('/data', express.static(path.join(__dirname, '..', 'data'), {
  index: false,
  dotfiles: 'ignore'
}));

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public'), {
  index: 'index.html',
  extensions: ['html'],
  dotfiles: 'ignore'
}));

// Explicit root route as fallback
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`server running at http://localhost:${PORT}`);
  console.log(`Press Ctrl+C to stop the server`);
});
