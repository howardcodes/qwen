/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./pages/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#080b12',
        foreground: '#f8fafc',
        card: '#111827',
        primary: '#7c3aed',
        accent: '#22d3ee'
      }
    },
  },
  plugins: [],
}
