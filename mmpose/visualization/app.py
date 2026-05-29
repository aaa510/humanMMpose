import time
from queue import Queue
from flask import Flask, request, render_template, Response, redirect, url_for
import cv2
import numpy as np
import threading

app = Flask(__name__)

# 创建一个队列来存储最新的帧
frame_queue = Queue(maxsize=30)  # 限制队列大小为30帧
frame_available = threading.Event()

def display_frames():
    """持续显示最新帧的线程函数"""
    while True:
        frame_available.wait()  # 等待新帧
        if not frame_queue.empty():
            frame = frame_queue.get()

            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break

@app.route('/')
def index():
    return redirect(url_for('video_feed'))  # 重定向到视频流页面
    """主页，渲染 HTML 表单"""


@app.route('/start_stream', methods=['POST'])
def start_stream():
    """接收并显示图像流"""
    try:
        print("收到请求")
        print(f"请求方法: {request.method}")
        print(f"请求文件: {request.files}")

        if 'image' not in request.files:
            print("未找到图像文件")
            return "No image found", 400

        # 获取并解码图像
        file = request.files['image']
        print(f"文件名: {file.filename}")

        # 读取文件数据
        file_data = file.read()
        np_img = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            print("图像解码失败")
            return "Invalid image data", 400

        # 将图像推送到队列中
        if not frame_queue.full():
            frame_queue.put(img)
            frame_available.set()  # 设置事件，表示有新的帧

        print("图像处理成功")
        return "Stream received", 200

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return f"Error processing image: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    """返回视频流"""

    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                # 将图像编码为 JPEG 格式并转换为字节流
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    # 生成视频流格式的响应，带有边界标识符
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # 使用 multipart/x-mixed-replace 来推送实时图像流
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(404)
def not_found(e):
    return "404 Not Found - The requested URL was not found on the server", 404

@app.errorhandler(405)
def method_not_allowed(e):
    return "405 Method Not Allowed - The method is not allowed for the requested URL", 405

def initialize_app():
    """初始化应用程序"""
    # 启动显示线程
    display_thread = threading.Thread(target=display_frames, daemon=True)
    display_thread.start()


if __name__ == '__main__':
    try:
        initialize_app()
        print("Server starting at http://localhost:5000")
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用缓存
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("正在关闭应用...")
    finally:
        # 清理资源
        cv2.destroyAllWindows()