const container = document.getElementById('container')
const fileInput = document.getElementById('file-input')

async function loadTrainingData() {
    const labels = [
        'Fukada Eimi',
        'Rina Ishihara',
        'Takizawa Laura',
        'Yua Mikami'
    ];
    const faceDescriptors = [];

    for (const label of labels) {
        const descriptors = [];
        for (let i = 1; i <= 4; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpeg`);
            const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
            
            if (detection && detection.descriptor) {
                descriptors.push(detection.descriptor);
            } else {
                console.warn(`Không thể phát hiện khuôn mặt hoặc descriptor bị thiếu trong ảnh: /data/${label}/${i}.jpeg`);
            }
        }
        
        if (descriptors.length > 0) {
            faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
            Toastify({ text: `Training xong dữ liệu của ${label}` }).showToast();
        } else {
            console.error(`Không có descriptor hợp lệ cho nhãn ${label}`);
        }
    }
    Toastify({ text: 'Đã train xong tất cả dữ liệu !!!' }).showToast();
    return faceDescriptors;
}

async function init() {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models')
    ]);
    Toastify({ text: 'Đã tải xong models nhận diện' }).showToast();

    const trainingData = await loadTrainingData();
    
    // Kiểm tra xem trainingData có được tải thành công không
    if (trainingData && trainingData.length > 0) {
        faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6);
    } else {
        console.error('Không thể khởi tạo faceMatcher do thiếu dữ liệu training hợp lệ.');
    }
}

init();


fileInput.addEventListener('change', async (e) => {
    const file = fileInput.files[0]

    const image = await faceapi.bufferToImage(file)
    const canvas = faceapi.createCanvasFromMedia(image)

    container.innerHTML = ''
    container.append(image)
    container.append(canvas)

    const size = {
        width: image.width,
        height: image.height
    }
    faceapi.matchDimensions(canvas, size)

    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    const resizeDetections = faceapi.resizeResults(detections, size)

    for (const detection of resizeDetections) {
        const box = detection.detection.box
        const drawBox = new faceapi.draw.DrawBox(box, {
            label: faceMatcher.findBestMatch(detection.descriptor)
        })
        drawBox.draw(canvas)
    }
})