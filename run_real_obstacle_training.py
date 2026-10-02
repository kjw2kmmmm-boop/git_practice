import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import os

# ==========================================
# 1. 장애물 탐지 전용 CNN 모델 (0: 빈공간, 1: 장애물)
# ==========================================
class ObstacleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv_layers(x)

# ==========================================
# 2. 대시보드(리포트) 렌더링 함수
# ==========================================
def create_obs_dashboard(pred_tensor, original_map, scale=4):
    h, w = original_map.shape
    vis_img = cv2.cvtColor(original_map, cv2.COLOR_GRAY2BGR)
    vis_img = cv2.resize(vis_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    pred_mask_255 = (pred_tensor * 255).astype(np.uint8)
    pred_resized = cv2.resize(pred_mask_255, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    blurred = cv2.GaussianBlur(pred_resized, (15, 15), 0)
    _, smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    smooth_contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in smooth_contours:
        if cv2.contourArea(cnt) < 150: 
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(vis_img, [approx], -1, (0, 0, 255), 3)

    legend_height = 150
    canvas_w = max(w * scale, 550)
    canvas_h = h * scale + legend_height
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    map_x_offset = (canvas_w - w * scale) // 2
    canvas[0:h*scale, map_x_offset:map_x_offset+w*scale] = vis_img

    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    
    try:
        font_title = ImageFont.truetype("malgun.ttf", 24)
        font_text = ImageFont.truetype("malgun.ttf", 16)
    except:
        font_title = font_text = ImageFont.load_default()

    start_y = h * scale + 20
    start_x = 40

    draw.text((start_x, start_y), "AI 실전 장애물 예측 리포트", font=font_title, fill=(0, 0, 0))
    draw.line([(start_x, start_y + 35), (canvas_w - 40, start_y + 35)], fill=(150, 150, 150), width=2)

    y_offset = start_y + 55
    draw.rectangle([start_x, y_offset, start_x + 25, y_offset + 25], outline=(255, 0, 0), width=3)
    draw.text((start_x + 40, y_offset + 2), "AI가 예측한 융합 장애물 윤곽 (Smooth)", font=font_text, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(script_dir, 'training_progress_obs')
    os.makedirs(output_dir, exist_ok=True)
    
    raw_map_path = os.path.join(script_dir, 'my_room_map.png')
    obs_map_path = os.path.join(script_dir, 'my_room_map_obs.png')
    
    raw_map = cv2.imread(raw_map_path, cv2.IMREAD_GRAYSCALE)
    obs_map = cv2.imread(obs_map_path, cv2.IMREAD_COLOR)

    if raw_map is None or obs_map is None:
        print("❌ 지도 파일을 찾을 수 없습니다.")
        exit()

    print("🔍 빨간색 장애물 윤곽 기반 정답지 추출 중...")
    red_mask = ((obs_map[:, :, 2] > 200) & (obs_map[:, :, 1] < 100) & (obs_map[:, :, 0] < 100)).astype(np.uint8)
    
    gt_obstacle_mask = np.zeros_like(red_mask)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_obstacle_mask, contours, -1, 1, thickness=cv2.FILLED)

    img_tensor = torch.tensor(raw_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    label_tensor = torch.tensor(gt_obstacle_mask, dtype=torch.long).unsqueeze(0)

    model = ObstacleCNN()
    
    # [수정 1] 초기 학습률을 0.005에서 0.001로 대폭 낮춰 안정성 확보
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # [수정 2] 스케줄러 도입: 오차가 1000번 에폭 동안 갱신되지 않으면 학습률을 절반(0.5)으로 깎음
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)
    
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 20000
    print(f"\n🚀 실전 장애물 예측 딥러닝 시작! (총 {EPOCHS} 에폭)")
    print(f"📁 결과물은 '{output_dir}' 폴더에 1000 에폭마다 저장됩니다.\n")

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        outputs = model(img_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
        
        # [수정 3] 스케줄러에게 현재 오차 상황 보고
        scheduler.step(loss)
        
        if epoch % 1000 == 0:
            # 현재 적용 중인 학습률(lr)을 확인하기 위해 출력에 추가
            current_lr = optimizer.param_groups[0]['lr']
            print(f" 🔄 에폭 {epoch:5d}/{EPOCHS} (Loss: {loss.item():.5f}) | LR: {current_lr:.6f} | 저장 완료")
            
            pred = torch.argmax(outputs, dim=1).squeeze(0).detach().numpy().astype(np.uint8)
            dashboard_img = create_obs_dashboard(pred, raw_map, scale=4)
            
            save_path = os.path.join(output_dir, f'obs_dashboard_epoch_{epoch:05d}.png')
            cv2.imwrite(save_path, dashboard_img)

    print(f"\n✅ 20000번 장기 학습 완료! 'training_progress_obs' 폴더를 열어보세요.")