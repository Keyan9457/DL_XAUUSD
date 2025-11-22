import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_workflow_diagram(filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define boxes
    boxes = [
        {'x': 0.5, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'Data Sources\n(MT5, YFinance)', 'color': '#e1f5fe'},
        {'x': 3.0, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'Preprocessing\n(Indicators, SMC)', 'color': '#fff9c4'},
        {'x': 5.5, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'AI Model\n(LSTM Prediction)', 'color': '#e8f5e9'},
        {'x': 8.0, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'Decision Logic\n(Waterfall Gates)', 'color': '#ffe0b2'},
        {'x': 10.5, 'y': 2.5, 'w': 1.5, 'h': 1, 'label': 'Execution\n(MT5 Order)', 'color': '#ffcdd2'}
    ]

    # Draw arrows
    for i in range(len(boxes) - 1):
        start_x = boxes[i]['x'] + boxes[i]['w']
        end_x = boxes[i+1]['x']
        y = boxes[i]['y'] + boxes[i]['h'] / 2
        ax.arrow(start_x, y, end_x - start_x - 0.1, 0, head_width=0.15, head_length=0.1, fc='k', ec='k')

    # Draw boxes
    for box in boxes:
        rect = patches.FancyBboxPatch((box['x'], box['y']), box['w'], box['h'], 
                                      boxstyle="round,pad=0.1", 
                                      linewidth=1, edgecolor='black', facecolor=box['color'])
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['label'], 
                ha='center', va='center', fontsize=10, fontweight='bold')

    plt.title('Project Workflow: From Data to Trade', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

def create_architecture_diagram(filename):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Define nodes for Waterfall Logic
    nodes = [
        {'x': 4, 'y': 10.5, 'w': 2, 'h': 1, 'label': 'Start Cycle', 'color': '#cfd8dc'},
        
        {'x': 4, 'y': 8.5, 'w': 2, 'h': 1, 'label': 'Gate 1: HTF Trend\n(30m & 15m EMA)', 'color': '#bbdefb'},
        
        {'x': 1.5, 'y': 6.5, 'w': 2, 'h': 1, 'label': 'Gate 2: AI Pred\n(Bullish)', 'color': '#c8e6c9'},
        {'x': 6.5, 'y': 6.5, 'w': 2, 'h': 1, 'label': 'Gate 2: AI Pred\n(Bearish)', 'color': '#ffcdd2'},
        
        {'x': 1.5, 'y': 4.5, 'w': 2, 'h': 1, 'label': 'Gate 3: LTF Entry\n(3m & 1m)', 'color': '#fff9c4'},
        {'x': 6.5, 'y': 4.5, 'w': 2, 'h': 1, 'label': 'Gate 3: LTF Entry\n(3m & 1m)', 'color': '#fff9c4'},
        
        {'x': 1.5, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'EXECUTE BUY', 'color': '#4caf50', 'text_color': 'white'},
        {'x': 6.5, 'y': 2.5, 'w': 2, 'h': 1, 'label': 'EXECUTE SELL', 'color': '#f44336', 'text_color': 'white'},
        
        {'x': 4, 'y': 0.5, 'w': 2, 'h': 0.8, 'label': 'HOLD / WAIT', 'color': '#eeeeee'}
    ]

    # Draw connections
    # Start -> Gate 1
    ax.arrow(5, 10.5, 0, -1, head_width=0.15, head_length=0.15, fc='k', ec='k')
    
    # Gate 1 -> Gate 2 (Bullish/Bearish)
    ax.arrow(5, 8.5, -2.5, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # To Bullish
    ax.arrow(5, 8.5, 2.5, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # To Bearish
    ax.arrow(5, 8.5, 0, -7.2, head_width=0.15, head_length=0.15, fc='gray', ec='gray', linestyle='--') # To Hold (Neutral)
    
    # Gate 2 -> Gate 3
    ax.arrow(2.5, 6.5, 0, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # Bullish -> LTF
    ax.arrow(7.5, 6.5, 0, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # Bearish -> LTF
    
    # Gate 3 -> Execution
    ax.arrow(2.5, 4.5, 0, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # LTF -> Buy
    ax.arrow(7.5, 4.5, 0, -1, head_width=0.15, head_length=0.15, fc='k', ec='k') # LTF -> Sell
    
    # Failures to Hold
    ax.plot([2.5, 2.5, 4], [6.5, 1.3, 1.3], color='gray', linestyle='--', alpha=0.5) # Gate 2 fail
    ax.plot([7.5, 7.5, 6], [6.5, 1.3, 1.3], color='gray', linestyle='--', alpha=0.5)
    
    # Draw nodes
    for node in nodes:
        rect = patches.FancyBboxPatch((node['x'], node['y']), node['w'], node['h'], 
                                      boxstyle="round,pad=0.1", 
                                      linewidth=1, edgecolor='black', facecolor=node['color'])
        ax.add_patch(rect)
        text_color = node.get('text_color', 'black')
        ax.text(node['x'] + node['w']/2, node['y'] + node['h']/2, node['label'], 
                ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)

    plt.title('System Architecture: The "Waterfall" Logic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

if __name__ == "__main__":
    create_workflow_diagram('workflow_diagram.png')
    create_architecture_diagram('architecture_diagram.png')
