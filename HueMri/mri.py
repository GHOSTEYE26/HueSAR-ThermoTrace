import os
import numpy as np
import cv2
from skimage import exposure
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinter import font as tkfont
import webbrowser
from tkinter import messagebox

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

class MedicalImageColorizer:
    def __init__(self, root):
        self.root = root
        self.root.title("HueSAR - Medical Imaging Suite")
        self.root.geometry("1600x900")
        
        # Theme variables
        self.is_dark_mode = False
        self.theme_colors = {
            'light': {
                'bg': '#f5f5f5',
                'fg': '#333333',
                'frame_bg': '#ffffff',
                'accent': '#2c7be5',
                'hover': '#1a5cb8',
                'pressed': '#0d4b9c',
                'trough': '#e0e0e0',
                'text': '#666666'
            },
            'dark': {
                'bg': '#1a1a1a',
                'fg': '#ffffff',
                'frame_bg': '#2d2d2d',
                'accent': '#4a90e2',
                'hover': '#357abd',
                'pressed': '#2a5d8a',
                'trough': '#404040',
                'text': '#cccccc'
            }
        }
        
        # Custom fonts
        self.title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.subtitle_font = tkfont.Font(family="Helvetica", size=14)
        self.label_font = tkfont.Font(family="Helvetica", size=12)
        self.button_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        
        # Variables
        self.input_image = None
        self.processed_image = None
        self.colormap_var = tk.StringVar(value="crystal")
        self.blend_var = tk.DoubleVar(value=0.3)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.enhance_var = tk.BooleanVar(value=True)
        self.mri_type_var = tk.StringVar(value="1.5T")
        self.modality_var = tk.StringVar(value="MRI")
        self.ct_type_var = tk.StringVar(value="standard")  # New variable for CT type
        
        # Configure styles
        self.configure_styles()
        self.setup_ui()
    
    def configure_styles(self):
        style = ttk.Style()
        current_theme = 'dark' if self.is_dark_mode else 'light'
        colors = self.theme_colors[current_theme]
        
        # Configure theme colors
        style.configure('Medical.TFrame', background=colors['bg'])
        style.configure('Medical.TLabelframe', 
                       background=colors['frame_bg'], 
                       borderwidth=2, 
                       relief='solid')
        style.configure('Medical.TLabelframe.Label', 
                       font=self.label_font, 
                       background=colors['frame_bg'],
                       foreground=colors['fg'])
        
        # Button styles
        style.configure('Medical.TButton', 
                      font=self.button_font,
                      padding=10,
                      background=colors['accent'],
                      foreground='white')
        style.map('Medical.TButton',
                 background=[('active', colors['hover']), 
                           ('pressed', colors['pressed'])])
        
        # Radiobutton styles
        style.configure('Medical.TRadiobutton',
                      font=self.label_font,
                      background=colors['frame_bg'],
                      foreground=colors['fg'])
        
        # Scale styles
        style.configure('Medical.Horizontal.TScale',
                      background=colors['frame_bg'],
                      troughcolor=colors['trough'])
        
        # Checkbutton styles
        style.configure('Medical.TCheckbutton',
                      font=self.label_font,
                      background=colors['frame_bg'],
                      foreground=colors['fg'])
        
        # Label styles
        style.configure('Medical.TLabel',
                      font=self.label_font,
                      background=colors['frame_bg'],
                      foreground=colors['fg'])
    
    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.configure_styles()
        self.update_theme_colors()
    
    def update_theme_colors(self):
        current_theme = 'dark' if self.is_dark_mode else 'light'
        colors = self.theme_colors[current_theme]
        
        # Update root background
        self.root.configure(bg=colors['bg'])
        
        # Update all frames and labels
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.configure(style='Medical.TFrame')
            elif isinstance(widget, ttk.Label):
                widget.configure(style='Medical.TLabel')
    
    def setup_ui(self):
        # Create main frame with padding and background
        main_frame = ttk.Frame(self.root, style='Medical.TFrame', padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame, 
                          bg=self.theme_colors['light' if not self.is_dark_mode else 'dark']['bg'],
                          width=1600,  # Set fixed width
                          height=900)  # Set fixed height
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Medical.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1600)  # Set window width
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header frame with logo and title
        header_frame = ttk.Frame(scrollable_frame, style='Medical.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 20))
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, 
                              text="HueSAR", 
                              font=self.title_font,
                              foreground=self.theme_colors['light' if not self.is_dark_mode else 'dark']['accent'])
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame,
                                 text="Advanced Medical Imaging Suite",
                                 font=self.subtitle_font,
                                 foreground=self.theme_colors['light' if not self.is_dark_mode else 'dark']['text'])
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        # Theme toggle button
        theme_button = ttk.Button(header_frame,
                                text="🌙 Dark Mode" if not self.is_dark_mode else "☀️ Light Mode",
                                command=self.toggle_theme,
                                style='Medical.TButton')
        theme_button.grid(row=0, column=1, sticky=tk.E, padx=10)
        
        # Modality selection frame with modern styling
        modality_frame = ttk.LabelFrame(scrollable_frame, 
                                      text="Select Imaging Modality",
                                      style='Medical.TLabelframe',
                                      padding="15")
        modality_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 20))
        
        # Modality selection buttons with modern styling
        mri_button = ttk.Radiobutton(modality_frame, 
                                   text="MRI Imaging",
                                   variable=self.modality_var, 
                                   value="MRI",
                                   style='Medical.TRadiobutton',
                                   command=self.update_modality_options)
        mri_button.grid(row=0, column=0, padx=20, pady=10)
        
        xray_button = ttk.Radiobutton(modality_frame, 
                                    text="X-ray Imaging",
                                    variable=self.modality_var, 
                                    value="X-ray",
                                    style='Medical.TRadiobutton',
                                    command=self.update_modality_options)
        xray_button.grid(row=0, column=1, padx=20, pady=10)
        
        ct_button = ttk.Radiobutton(modality_frame, 
                                  text="CT Scan",
                                  variable=self.modality_var, 
                                  value="CT",
                                  style='Medical.TRadiobutton',
                                  command=self.update_modality_options)
        ct_button.grid(row=0, column=2, padx=20, pady=10)
        
        # Left panel for controls
        self.control_frame = ttk.LabelFrame(scrollable_frame, 
                                          text="Image Controls",
                                          style='Medical.TLabelframe',
                                          padding="20")
        self.control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # File selection button with modern styling
        self.select_button = ttk.Button(self.control_frame, 
                                      text="Select Medical Image", 
                                      command=self.select_image,
                                      style='Medical.TButton')
        self.select_button.grid(row=0, column=0, columnspan=2, pady=15, sticky=tk.EW)
        
        # MRI specific options frame
        self.mri_frame = ttk.LabelFrame(self.control_frame, 
                                      text="MRI Settings",
                                      style='Medical.TLabelframe',
                                      padding="15")
        self.mri_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Radiobutton(self.mri_frame, 
                       text="1.5T MRI (Standard)",
                       variable=self.mri_type_var, 
                       value="1.5T",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(self.mri_frame, 
                       text="3T MRI (Advanced)",
                       variable=self.mri_type_var, 
                       value="3T",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        # X-ray specific options frame
        self.xray_frame = ttk.LabelFrame(self.control_frame, 
                                       text="X-ray Settings",
                                       style='Medical.TLabelframe',
                                       padding="15")
        self.xray_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Radiobutton(self.xray_frame, 
                       text="Standard X-ray",
                       variable=self.mri_type_var, 
                       value="standard",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(self.xray_frame, 
                       text="High Resolution X-ray",
                       variable=self.mri_type_var, 
                       value="highres",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        # CT specific options frame
        self.ct_frame = ttk.LabelFrame(self.control_frame, 
                                     text="CT Settings",
                                     style='Medical.TLabelframe',
                                     padding="15")
        self.ct_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Radiobutton(self.ct_frame, 
                       text="Standard CT",
                       variable=self.ct_type_var, 
                       value="standard",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(self.ct_frame, 
                       text="High Resolution CT",
                       variable=self.ct_type_var, 
                       value="highres",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(self.ct_frame, 
                       text="Low Dose CT",
                       variable=self.ct_type_var, 
                       value="lowdose",
                       style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        # Colormap selection with modern styling
        self.colormap_frame = ttk.LabelFrame(self.control_frame, 
                                           text="Colormap Options",
                                           style='Medical.TLabelframe',
                                           padding="15")
        self.colormap_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Initialize colormap options
        self.update_colormap_options()
        
        # Blend factor slider with modern styling
        blend_frame = ttk.LabelFrame(self.control_frame, 
                                   text="Blend Settings",
                                   style='Medical.TLabelframe',
                                   padding="15")
        blend_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(blend_frame, 
                 text="Original Image Blend:",
                 font=self.label_font).pack(anchor=tk.W)
        blend_scale = ttk.Scale(blend_frame, 
                              from_=0, 
                              to=1, 
                              variable=self.blend_var, 
                              orient=tk.HORIZONTAL, 
                              length=200,
                              style='Medical.Horizontal.TScale')
        blend_scale.pack(fill=tk.X, pady=5)
        
        # Gamma correction slider with modern styling
        gamma_frame = ttk.LabelFrame(self.control_frame, 
                                   text="Image Enhancement",
                                   style='Medical.TLabelframe',
                                   padding="15")
        gamma_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(gamma_frame, 
                 text="Gamma Correction:",
                 font=self.label_font).pack(anchor=tk.W)
        gamma_scale = ttk.Scale(gamma_frame, 
                              from_=0.1, 
                              to=3.0, 
                              variable=self.gamma_var, 
                              orient=tk.HORIZONTAL, 
                              length=200,
                              style='Medical.Horizontal.TScale')
        gamma_scale.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(gamma_frame, 
                       text="Enhance Contrast",
                       variable=self.enhance_var, 
                       style='Medical.TCheckbutton').pack(anchor=tk.W, pady=5)
        
        # Process button with modern styling
        process_button = ttk.Button(self.control_frame, 
                                  text="Process Image", 
                                  command=self.process_image,
                                  style='Medical.TButton')
        process_button.grid(row=5, column=0, columnspan=2, pady=20, sticky=tk.EW)
        
        # Right panel for image display with modern styling
        self.image_frame = ttk.LabelFrame(scrollable_frame, 
                                        text="Image Preview",
                                        style='Medical.TLabelframe',
                                        padding="20")
        self.image_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Create image labels with modern styling
        original_frame = ttk.LabelFrame(self.image_frame, 
                                      text="Original Image",
                                      style='Medical.TLabelframe',
                                      padding="15")
        original_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        processed_frame = ttk.LabelFrame(self.image_frame, 
                                       text="Processed Image",
                                       style='Medical.TLabelframe',
                                       padding="15")
        processed_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NSEW)
        self.processed_label = ttk.Label(processed_frame)
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        # Configure grid weights and styles
        scrollable_frame.columnconfigure(1, weight=1)
        scrollable_frame.rowconfigure(2, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)
        
        # Bind mousewheel for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Initialize modality-specific options
        self.update_modality_options()
    
    def update_modality_options(self):
        # Show/hide modality-specific frames
        if self.modality_var.get() == "MRI":
            self.mri_frame.grid()
            self.xray_frame.grid_remove()
            self.ct_frame.grid_remove()
            self.select_button.configure(text="Select MRI Image")
        elif self.modality_var.get() == "X-ray":
            self.mri_frame.grid_remove()
            self.xray_frame.grid()
            self.ct_frame.grid_remove()
            self.select_button.configure(text="Select X-ray Image")
        else:  # CT
            self.mri_frame.grid_remove()
            self.xray_frame.grid_remove()
            self.ct_frame.grid()
            self.select_button.configure(text="Select CT Image")
        
        # Update colormap options based on modality
        self.update_colormap_options()

    def update_colormap_options(self):
        # Clear existing colormap options
        for widget in self.colormap_frame.winfo_children():
            widget.destroy()
        
        # Add modality-specific colormaps
        if self.modality_var.get() == "MRI":
            colormaps = ['crystal', 'medical', 'bone', 'cool', 'viridis', 'plasma']
            description = "Crystal: Enhanced detail visibility for nerves and soft tissues"
        elif self.modality_var.get() == "X-ray":
            colormaps = ['crystal', 'bone', 'gray', 'hot', 'copper']
            description = "Crystal: Enhanced detail visibility for bones and structures"
        else:  # CT
            colormaps = ['crystal', 'bone', 'gray', 'hot', 'copper', 'bone_enhanced']
            description = "Crystal: Enhanced detail visibility for soft tissues and bones"
        
        for cmap in colormaps:
            ttk.Radiobutton(self.colormap_frame, text=cmap.capitalize(), variable=self.colormap_var, 
                          value=cmap, style='Medical.TRadiobutton').pack(anchor=tk.W, pady=5)
        
        # Add description for crystal colormap
        crystal_desc = ttk.Label(self.colormap_frame, 
                               text=description,
                               font=('Helvetica', 9, 'italic'),
                               wraplength=200)
        crystal_desc.pack(anchor=tk.W, pady=(0, 10))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Medical Images", "*.png *.jpg *.jpeg *.tif *.tiff *.dcm")]
        )
        if file_path:
            self.input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.input_image is not None:
                self.display_image(self.input_image, self.original_label, "Original")
    
    def display_image(self, image, label, title):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        max_size = 500  # Increased size for better visibility
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = cv2.resize(image, (new_width, new_height))
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        label.configure(image=photo)
        label.image = photo
    
    def process_image(self):
        if self.input_image is None:
            return
        
        # Apply modality-specific processing
        if self.modality_var.get() == "MRI":
            # MRI processing
            if self.mri_type_var.get() == "3T":
                processed_image = self.enhance_contrast(self.input_image, clip_limit=0.05)
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get() * 0.9)
            else:
                if self.enhance_var.get():
                    processed_image = self.enhance_contrast(self.input_image)
                else:
                    processed_image = self.input_image
                
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get())
        elif self.modality_var.get() == "X-ray":
            # X-ray processing
            if self.mri_type_var.get() == "highres":
                processed_image = self.enhance_contrast(self.input_image, clip_limit=0.03)
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get() * 0.7)
            else:
                if self.enhance_var.get():
                    processed_image = self.enhance_contrast(self.input_image, clip_limit=0.02)
                else:
                    processed_image = self.input_image
                
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get())
        else:  # CT processing
            if self.ct_type_var.get() == "highres":
                processed_image = self.enhance_contrast(self.input_image, clip_limit=0.04)
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get() * 0.8)
            elif self.ct_type_var.get() == "lowdose":
                processed_image = self.enhance_contrast(self.input_image, clip_limit=0.06)
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get() * 1.2)
            else:  # standard
                if self.enhance_var.get():
                    processed_image = self.enhance_contrast(self.input_image, clip_limit=0.03)
                else:
                    processed_image = self.input_image
                
                if self.gamma_var.get() != 1.0:
                    processed_image = self.apply_gamma_correction(processed_image, self.gamma_var.get())
        
        # Apply colormap
        colored_image = self.apply_colormap(processed_image, self.colormap_var.get())
        
        # Blend with original
        if self.blend_var.get() > 0:
            final_image = self.blend_with_original(processed_image, colored_image, self.blend_var.get())
        else:
            final_image = colored_image
        
        self.display_image(final_image, self.processed_label, "Processed")
    
    def enhance_contrast(self, image, clip_limit=0.03):
        img_float = img_as_float(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_as_ubyte(img_float))
        return enhanced
    
    def create_custom_medical_colormap(self):
        colors = [(0, 0, 0.3),      # Dark blue for low values
                  (0, 0.5, 0.8),    # Blue
                  (0, 0.8, 0.8),    # Cyan
                  (0.2, 0.8, 0.2),  # Green
                  (0.8, 0.8, 0),    # Yellow
                  (0.8, 0.4, 0),    # Orange
                  (0.8, 0, 0)]      # Red for high values
        return LinearSegmentedColormap.from_list('medical', colors)

    def create_crystal_colormap(self):
        # Enhanced colormap for better detail visibility
        colors = [
            (0.0, 0.0, 0.0),        # Pure black for background
            (0.1, 0.1, 0.3),        # Dark blue for low intensity
            (0.2, 0.4, 0.8),        # Bright blue for soft tissues
            (0.4, 0.8, 0.9),        # Cyan for enhanced soft tissue contrast
            (0.6, 0.9, 0.6),        # Light green for medium intensity
            (0.8, 0.9, 0.4),        # Yellow for bone structures
            (0.9, 0.7, 0.2),        # Orange for enhanced bone details
            (1.0, 0.5, 0.0),        # Red for high intensity areas
            (1.0, 1.0, 1.0)         # White for maximum intensity
        ]
        return LinearSegmentedColormap.from_list('crystal', colors)

    def apply_colormap(self, image, colormap_name='medical'):
        normalized = img_as_float(image)
        
        # Apply additional contrast enhancement for crystal colormap
        if colormap_name == 'crystal':
            # Enhance local contrast
            clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_as_ubyte(normalized))
            normalized = img_as_float(enhanced)
            
            # Apply gamma correction for better detail visibility
            normalized = exposure.adjust_gamma(normalized, 0.8)
            
            # Normalize again after enhancements
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        
        if colormap_name == 'medical':
            cmap = self.create_custom_medical_colormap()
        elif colormap_name == 'crystal':
            cmap = self.create_crystal_colormap()
        else:
            cmap = plt.get_cmap(colormap_name)
        
        colored = cmap(normalized)
        colored_bgr = cv2.cvtColor((colored[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return colored_bgr
    
    def blend_with_original(self, original, colored, blend_factor=0.3):
        original_3ch = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(original_3ch, blend_factor, colored, 1.0 - blend_factor, 0)
        return blended
    
    def apply_gamma_correction(self, image, gamma=1.0):
        if gamma == 1.0:
            return image
        return exposure.adjust_gamma(image, gamma)

def main():
    root = tk.Tk()
    app = MedicalImageColorizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()