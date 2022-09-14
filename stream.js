let $ = require('jquery')  // jQuery now loaded and assigned to $

var img = document.createElement("img");
img.src = document.getElementById("image-backend");

var block = document.getElementById("image");


$('#image').html(block)
