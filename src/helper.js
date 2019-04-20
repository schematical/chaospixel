var Jimp = require('jimp');
var async = require("async");
const X_OFFSET = 256;
const TILE_SIZE = 16;
const MAP_WIDTH = 4352;
const MAP_HEIGHT = 1408
const ROOM_WIDTH = (MAP_WIDTH - X_OFFSET) / 16;//16 * TILE_SIZE;
const ROOM_HEIGHT = MAP_HEIGHT / 8;//11 * TILE_SIZE;
const DEST_PATH = '../data/zelda_map/';
// open a file called "lenna.png"
Jimp.read('../map/zelda1.png', (err, image) => {
  if (err) throw err;
    let crops = [];
    for(let x = 0; x < 16; x++){
        for(let y = 0; y < 8; y++){
            let crop = {
                x: (x * ROOM_WIDTH) + X_OFFSET,
                y: y * ROOM_HEIGHT,
                w: ROOM_WIDTH,
                h: ROOM_HEIGHT,
                dest: DEST_PATH + x + '-' + y +'.png'
            }
            crops.push(crop);

        }
    }

    //Make tiles
    for(let x = 0; x < MAP_WIDTH; x +=  TILE_SIZE){
        for(let y = 0; y < MAP_HEIGHT; y +=  TILE_SIZE){
            let crop = {
                x: (x) + X_OFFSET
                y: y,
                w: TILE_SIZE,
                h: TILE_SIZE,
                dest: '../data/zelda_tiles/' + x + '-' + y +'.png'
            }
        }

    }
    async.eachSeries(crops, function iteratee(crop, callback) {
        console.log("Starting: " + crop.dest);
        image.clone()
            .crop(crop.x , crop.y, crop.w,  crop.h)
            .write(crop.dest,callback);

    }, function done(err) {
        //...
        if(err){
            throw err;
        }
        console.log("DONE");
    });


});