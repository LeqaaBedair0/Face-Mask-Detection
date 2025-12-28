const API = "http://127.0.0.1:8000";
let stream = null;

// رفع صورة
document.getElementById('imageUpload').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;

    document.getElementById('imagePreview').innerHTML =
        `<img src="${URL.createObjectURL(file)}" style="max-width:100%;border-radius:12px;margin-top:15px;">`;

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch(`${API}/api/predict`, { method: 'POST', body: form });
        const data = await res.json();

        const r = document.getElementById('result');
        r.className = `result ${data.prediction === 'with_mask' ? 'with-mask' : 'without-mask'}`;
        r.innerHTML = data.prediction === 'with_mask'
            ? `<strong>يلبس كمامة</strong> – الثقة: ${data.confidence}%`
            : `<strong>لا يلبس كمامة</strong> – الثقة: ${data.confidence}%`;
    } catch (err) {
        console.error("خطأ في رفع الصورة:", err);
    }
};

// الكاميرا – تصنيف مباشر
document.getElementById('startCamera').onclick = async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        const video = document.getElementById('video');
        video.srcObject = stream;

        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('stopCamera').style.display = 'inline-block';

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        const loop = async () => {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                try {
                    const res = await fetch(`${API}/api/realtime`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.6) })
                    });

                    if (res.ok) {
                        const d = await res.json();

                        // عرض النتيجة في الـ div تحت الفيديو
                        const rr = document.getElementById('realtimeResult');
                        rr.className = d.prediction === 'with_mask' ? 'with-mask' : 'without-mask';
                        rr.innerHTML = d.prediction === 'with_mask'
                            ? `<strong>مع كمامة</strong> – الثقة: ${d.confidence}%`
                            : `<strong>بدون كمامة</strong> – الثقة: ${d.confidence}%`;

                        // عرض النتيجة مكتوبة فوق الفيديو
                        ctx.fillStyle = d.prediction === 'with_mask' ? '#00ff00' : '#ff0000';
                        ctx.font = 'bold 40px Cairo';
                        ctx.fillText(
                            d.prediction === 'with_mask' ? 'مع كمامة' : 'بدون كمامة',
                            50, 50
                        );
                        ctx.fillText(`${d.confidence}%`, 50, 100);
                    }
                } catch (err) {
                    console.error("خطأ في التصنيف المباشر:", err);
                }
            }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);

    } catch (err) {
        alert("مش قادر يفتح الكاميرا – تأكدي إنك سمحتي للمتصفح");
    }
};

// إيقاف الكاميرا
document.getElementById('stopCamera').onclick = () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
    document.getElementById('video').srcObject = null;
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('stopCamera').style.display = 'none';
};