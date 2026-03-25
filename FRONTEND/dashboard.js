const uploadimg = document.querySelector(".uploadimg");
const uploadimgc = document.querySelector(".uploadimgc");
const input1 = document.querySelector("#input1");
const img1 = document.querySelector(".img1");
const uploadbtn = document.querySelector(".uploadbtn");
const firstb1 = document.querySelector(".firstb1");
const results = document.querySelector(".results");
const model = document.querySelector(".model");
const showmodels = document.querySelector(".showmodels");
const showmodelswrapper = document.querySelector(".showmodelswrapper");
const heading = document.querySelector(".heading");
let modelSelected = null;

document.addEventListener("click", (e) => {
    changemodel(e);
})

function changemodel(e) {
    // console.log(showmodelswrapper)
    // showmodelswrapper.querySelectorAll(".aboutmodel").forEach(element => {
    //     console.log(element);
    if (!model.contains(e.target) && !showmodels.contains(e.target) && !showmodelswrapper.contains(e.target)) {
        showmodels.style.display = "none";
        return;
    }
    // });
    // console.log(showmodelswrapper.querySelectorAll(".aboutmodel"))

    showmodelswrapper.querySelectorAll(".aboutmodel").forEach(element => {
        // console.log(element);
        if (element.querySelector(".model1name").innerText === model.innerText && !model.contains(e.target)) {
            showmodels.style.display = "none";
            return;
        }
        if (element.contains(e.target)) {
            modelSelected = element.querySelector(".model1name").innerText;
            model.innerText = modelSelected;
            heading.innerText = `Upload image to check the dog and cat(${model.innerText}).`
            showmodels.style.display = "none";
        }
    })

}

model.addEventListener("click", () => {
    showmodels.style.display = "block";
})

results.addEventListener("click", () => {
    window.location.href = "results.html";
})

firstb1.addEventListener("click", () => {
    window.location.href = `about.html`;
})



input1.addEventListener("change", uploadimginp);
function uploadimginp() {
    let imglink = URL.createObjectURL(input1.files[0]);
    img1.src = `${imglink}`;
    // uploadimg.style.backgroundImage = `url(${imglink})`;
    img1.style.display = "block";
    // uploadimg.style.display = "none";
    uploadimgc.style.display = "none";
    uploadimg.style.backgroundColor = "rgb(228, 228, 228)";
}


uploadbtn.addEventListener("click", async () => {
    try {
        uploadimginp();
        if(!input1.files[0]) return;
        uploadbtn.innerText = "Predicting...";

        let model_name = model.innerText
        

        if(model_name === "Switch model"){
            model_name = "EfficientNet"
        }
        const formdata = new FormData()

        formdata.append("file",input1.files[0])
        formdata.append("model_name",model_name)

        const result = await fetch("http://127.0.0.1:8000/Comparison", {
            method: "POST",
            body: formdata
        })
        const data = await result.json()

        const answer = document.querySelector(".answer");
        const answer1 = document.querySelector(".answer1");

        answer.style.display = "block";
        answer1.style.display = "block";
        
        answer.innerText = `The Animal present in image is ${data.pred_label}.`
        answer1.innerText = `Model confidence score is ${data.conf.toFixed(3)}.`
        uploadbtn.innerText = "Upload Image";
    }
    catch (err) {
        console.log(err);
        alert("Internal Server Error");
        return;
    }
})

uploadimg.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadimg.style.backgroundColor = "rgb(205, 205, 205)";
})
uploadimg.addEventListener("drop", (e) => {
    e.preventDefault();
    input1.files = e.dataTransfer.files;
    uploadimginp();
})





