const {app, BrowserWindow} = require('electron')
const url = require('url')
const path = require('path')

let win

function createWindow() {
   mainWindow = new BrowserWindow({width: 800, height: 600})
   mainWindow.loadURL("http://localhost:1000/")
}

app.on('ready', createWindow)