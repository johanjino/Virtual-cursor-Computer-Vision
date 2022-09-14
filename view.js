let $ = require('jquery')  // jQuery now loaded and assigned to $
let status = "Turned Off"
$('#cursor-status').text(status)
$('#start-cursor').on('click', () => {
   count = "Working..."
   $('#cursor-value').text(status)
}) 