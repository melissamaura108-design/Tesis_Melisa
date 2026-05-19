"""
evaluate.py - Evaluación del modelo entrenado con métricas detalladas
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataset_loader import get_data_loaders
from model import create_model
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Dispositivo de evaluación: {device}")

def evaluate_model(model_path="modelo_impresion.pth"):
    """
    Evalúa el modelo entrenado y muestra métricas detalladas
    
    Args:
        model_path: ruta al archivo de pesos del modelo
    """
    
    print("📂 Cargando dataset de validación...")
    _, val_loader, class_names = get_data_loaders(batch_size=32)
    
    num_classes = len(class_names)
    print(f"📊 Clases: {class_names}")
    
    print("🏗️ Cargando modelo...")
    model = create_model(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Modelo cargado desde {model_path}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔄 Evaluando modelo...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Obtener predicciones y probabilidades
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Convertir a arrays numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("=" * 60)
    
    print(f"\n✅ Precisión Global: {accuracy*100:.2f}%")
    
    print("\n📈 Reporte de Clasificación:")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n📊 Matriz de Confusión:")
    print("-" * 60)
    print("Filas: Valor Real | Columnas: Predicción")
    print("   " + "  ".join([f"{c[:3]}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i][:3]:3} {row}")
    
    # Mostrar matriz de confusión gráfica
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión - Detección de Fallos Impresión 3D')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig('matriz_confusion.png', dpi=150)
        print("\n📸 Matriz de confusión guardada como: matriz_confusion.png")
        plt.show()
    except Exception as e:
        print(f"⚠️ No se pudo mostrar gráfico: {e}")
    
    # Análisis por clase
    print("\n📊 Análisis por Clase:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        class_mask = (all_labels == i)
        class_correct = np.sum(all_preds[class_mask] == i)
        class_total = np.sum(class_mask)
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        
        # Probabilidad promedio para esta clase
        avg_prob = np.mean(all_probs[class_mask][:, i]) if class_total > 0 else 0
        
        print(f"  {class_name:15s}: {class_acc:.1f}% ({class_correct}/{class_total}) | Confianza media: {avg_prob:.2%}")
    
    # Detectar clases con bajo rendimiento
    print("\n⚠️ Advertencias:")
    for i, class_name in enumerate(class_names):
        class_mask = (all_labels == i)
        class_total = np.sum(class_mask)
        if class_total > 0:
            class_correct = np.sum(all_preds[class_mask] == i)
            class_acc = 100 * class_correct / class_total
            if class_acc < 70:
                print(f"  ⚠️ Clase '{class_name}' tiene bajo rendimiento: {class_acc:.1f}%")
                print(f"     Considerar agregar más imágenes de esta clase")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }

def evaluate_on_test_images(model_path="modelo_impresion.pth", image_paths=None):
    """
    Evalúa el modelo en imágenes específicas
    
    Args:
        model_path: ruta al modelo
        image_paths: lista de rutas de imágenes a evaluar
    """
    from PIL import Image
    from torchvision import transforms
    
    if image_paths is None:
        return
    
    # Transformaciones para inferencia
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar modelo
    _, _, class_names = get_data_loaders(batch_size=32)
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n🔍 Evaluando imágenes específicas:")
    print("-" * 60)
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            class_name = class_names[predicted.item()]
            confidence_val = confidence.item()
            
            print(f"\n  📷 {img_path}")
            print(f"     Predicción: {class_name}")
            print(f"     Confianza: {confidence_val:.2%}")
            
            # Mostrar top-3 predicciones
            top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
            print(f"     Top-3: ", end="")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                print(f"{class_names[idx]}: {prob.item():.1%}", end="  " if i < 2 else "")
            print()
            
        except Exception as e:
            print(f"  ❌ Error con {img_path}: {e}")

if __name__ == "__main__":
    evaluate_model()
    
    # Ejemplo de evaluación en imágenes específicas (descomentar para usar)
    # test_images = ["test1.jpg", "test2.jpg"]
    # evaluate_on_test_images(image_paths=test_images)