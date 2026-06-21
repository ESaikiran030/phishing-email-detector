async function detectEmail(){

let email=document.getElementById("email").value;

if(email.trim()===""){
alert("Enter Email Content");
return;
}

let formData=new FormData();

formData.append("email",email);

let response=await fetch("/predict",{
method:"POST",
body:formData
});

let data=await response.json();

let result=document.getElementById("result");

result.style.display="block";

if(data.prediction==="phishing"){

result.className="phishing";

result.innerHTML=`
⚠️ PHISHING EMAIL DETECTED<br><br>
Confidence : ${data.confidence}%<br>
URLs Found : ${data.urls}<br>
Suspicious Keywords : ${data.keywords}
`;
}
else{

result.className="safe";

result.innerHTML=`
✅ SAFE EMAIL<br><br>
Confidence : ${data.confidence}%<br>
URLs Found : ${data.urls}<br>
Suspicious Keywords : ${data.keywords}
`;
}
}