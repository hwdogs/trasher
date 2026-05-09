import os
import sys
import traceback


def test_step_1():
    """测试基础导入"""
    print("Step 1: 导入基础库...")
    import PIL.Image as Image
    print("✅ PIL 导入成功")

def test_step_2():
    """测试 Matplotlib 导入（无 GUI）"""
    print("Step 2: 导入 Matplotlib...")
    import matplotlib.pyplot as plt
    print("✅ Matplotlib.pyplot 导入成功")

def test_step_3(dataset_root_path):
    """测试读取第一张图片"""
    import PIL.Image as Image
    print("Step 3: 读取第一张图片...")
    for root, dirs, files in os.walk(dataset_root_path):
        if files:
            first_file = os.path.join(root, files[0])
            print(f"尝试打开: {first_file}")
            img = Image.open(first_file)
            size = img.size
            print(f"✅ 图片读取成功，尺寸: {size}")
            img.close()
            return
    print("❌ 未找到图片")

def test_step_4(dataset_root_path):
    """测试绘图（不显示）"""
    print("Step 4: 测试绘图...")
    import matplotlib.pyplot as plt
    plt.plot([1,2,3], [4,5,6])
    plt.savefig('test_plot.png')
    plt.close()
    print("✅ 绘图保存成功")

if __name__ == '__main__':
    dataset_root_path = r"..\dataset\dataset_55_4"
    
    steps = [test_step_1, test_step_2, 
             lambda: test_step_3(dataset_root_path), 
             lambda: test_step_4(dataset_root_path)]
    
    for i, step in enumerate(steps):
        try:
            step()
        except Exception as e:
            print(f"❌ 步骤 {i+1} 失败: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("🎉 所有测试通过！")
    input("按回车退出...")