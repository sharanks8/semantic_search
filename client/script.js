let fileinput = document.getElementById("fileinput");
let searchbutton = document.getElementById("search-button")
searchbutton.addEventListener('click',search)
let file;
fileinput.addEventListener('change',upload)
let json;
var outputCanvas = document.createElement('canvas');
var ctx = outputCanvas.getContext('2d');
// console.log('heree')


async function upload(){
    file = fileinput.files[0]
    const reader = new FileReader()
    reader.onload = function(e){
        var img = new Image();
                img.src = e.target.result;

                img.onload = function () {
                    
                    document.body.appendChild(outputCanvas);
                    outputCanvas.width = img.width;
                    outputCanvas.height = img.height;

                    
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                };
            };

            reader.readAsDataURL(file);
            
            
        
    }
async function search(){
    
    bbxs = await call_api();
        
            for(i = 0; i<bbxs.length;i++){
                
                x = bbxs[i][0]
                y = bbxs[i][1]
                w = bbxs[i][2] 
                h = bbxs[i][3] 
                console.log(x,y,w,h)
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
                // ctx.fillRect(0, 0, 20, 20);
                ctx.fillRect(x, y, w, h);


                           }


}

 async function call_api(){
    const data = new FormData();
    query = document.getElementById('search-box').value
    console.log(query)
    data.append('image', file);
    data.append('query', query);
    url = 'http://127.0.0.1:8081/search'
    let response = await fetch(url, {  
		// mode: 'no-cors',
		method: 'POST',
		body: data // send POST data
	  }); 
    if (response.ok) { // if HTTP-status is 200-299 
    // get the response body (the method explained below) 
    let json = await response.json();
  
    
    return json['bbx']
                        } 
 

    else { alert("HTTP-Error: " + response.status); 
    }

}
