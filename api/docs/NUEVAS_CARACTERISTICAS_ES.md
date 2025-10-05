# Nuevas Características del Modelo de Predicción de Blooms

## Resumen

Se han implementado exitosamente las características solicitadas para mejorar el modelo de predicción de blooms. El modelo ahora incluye 10 nuevas características ecológicas basadas en coordenadas geográficas y datos ambientales.

## Características Implementadas

### 1. Fecha de Inicio de Primavera (4 características)

**Implementación según especificaciones:**

✅ **a) Suavizado de datos NDVI con promedio móvil de 5 días**
- Elimina inconsistencias diarias
- Crea una curva más limpia y estable
- Muestra la tendencia real de crecimiento

✅ **b) Cálculo del promedio NDVI invernal**
- Período: 1 de diciembre a 1 de marzo
- Establece la línea base de vegetación dormante

✅ **c) Detección del inicio de primavera**
- La primavera comienza cuando el NDVI suavizado supera el promedio invernal
- Se mantiene arriba con el aumento sostenido más largo del año
- Detecta automáticamente el período de crecimiento más significativo

**Características generadas:**
- `spring_start_day`: Día del año cuando inicia la primavera
- `days_since_spring_start`: Días transcurridos desde el inicio de primavera
- `is_spring_active`: Indicador si actualmente estamos en primavera
- `winter_ndvi_baseline`: Promedio NDVI invernal

### 2. Grados Día de Crecimiento - GDD (2 características)

**Implementación según método BE (Baskerville y Emin, 1969):**

✅ **Fórmula implementada:**
```
GDD = [(Tmax + Tmin) / 2] - Tbase
donde Tbase = 0°C
```

✅ **Características generadas:**
- `gdd_current`: GDD del día actual
- `gdd_accumulated_30d`: GDD acumulados en los últimos 30 días

**Significado:** Mayor GDD = mayor potencial de crecimiento y desarrollo más rápido

### 3. Días de Agua Disponible en el Suelo (4 características)

**Implementación según método del Punto de Marchitez Permanente (PMP):**

✅ **Cálculo del punto de marchitez:**
```
PMP% = (CC% × 0.74) - 5
```
donde CC es la capacidad del campo

✅ **Cálculo de agua disponible:**
```
Si humedad_suelo < PMP:
    días_agua_suelo = 0 (estrés hídrico)
Sino:
    días_agua_suelo = humedad_suelo - PMP
```

✅ **Características generadas:**
- `soil_water_days`: Días de agua disponible para las plantas
- `wilting_point`: Punto de marchitez calculado
- `water_stress`: Indicador de estrés hídrico
- `available_water_ratio`: Relación de agua disponible (0-1)

**Nota:** El sistema considera que el punto de marchitez varía entre especies (como se especificó para plantas con hojas de aguja como los cactus).

## Archivos Creados/Modificados

### Nuevos Archivos:

1. **`api/app/bloom_features.py`**
   - Módulo completo con todas las funciones de cálculo
   - Funciones independientes que se pueden usar por separado
   - Incluye ejemplos y pruebas

2. **`api/test_new_features.py`**
   - Suite de pruebas completa
   - Valida todos los cálculos
   - Verifica integración con el modelo

3. **`api/ADVANCED_FEATURES_README.md`**
   - Documentación técnica completa en inglés
   - Referencias científicas
   - Guías de uso

### Archivos Modificados:

1. **`api/app/earth_engine_utils.py`**
   - Agregadas funciones para obtener series temporales de NDVI
   - Agregadas funciones para obtener datos de temperatura
   - Agregadas funciones para obtener humedad del suelo

2. **`api/app/bloom_predictor_v2.py`**
   - Integradas las nuevas características
   - Actualizado de 21 a 31 características totales
   - Modificados métodos de predicción

## Cómo Usar

### 1. Probar las Nuevas Características

```bash
cd api
python test_new_features.py
```

**Resultado esperado:** Todos los tests pasan ✓

### 2. Reentrenar el Modelo

```bash
cd api
python retrain_and_save_v2.py
```

Esto entrenará el modelo con las 31 características (21 originales + 10 nuevas).

### 3. Usar las Características Directamente

```python
from app.bloom_features import calculate_comprehensive_bloom_features

env_data = {
    'ndvi_time_series': [datos_ndvi_90_dias],
    'dates': [fechas_correspondientes],
    'tmax': [temperaturas_maximas],
    'tmin': [temperaturas_minimas],
    'soil_moisture': 22,
    'field_capacity': 25
}

features = calculate_comprehensive_bloom_features(env_data)
```

## Fuentes de Datos

### Con Google Earth Engine (Recomendado):

1. **NDVI:** MODIS/061/MOD13Q1 (250m, composición cada 16 días)
2. **Temperatura:** MODIS/061/MOD11A1 (1km, diario)
3. **Humedad del Suelo:** NASA/SMAP/SPL4SMGP/007 (10km, cada 3 horas)

### Modo Fallback (Sin Earth Engine):

El sistema genera datos sintéticos basados en:
- Modelos climáticos dependientes de latitud/longitud
- Patrones estacionales
- Variación espacial

## Resultados de Pruebas

```
✓ Detección de inicio de primavera: PASADO
✓ Cálculo de GDD: PASADO  
✓ Disponibilidad de agua en suelo: PASADO
✓ Integración de características: PASADO
✓ Integración con modelo: REQUIERE REENTRENAMIENTO
```

## Mejoras Esperadas

Las nuevas características deberían mejorar la precisión del modelo en:

1. **Timing Estacional** (características de primavera)
   - Captura el inicio real de primavera vs. fechas de calendario
   - Detecta primaveras tempranas vs. tardías

2. **Acumulación de Calor** (GDD)
   - Modela el desarrollo basado en energía térmica
   - Predice floraciones más tempranas en años cálidos

3. **Impacto del Estrés Hídrico** (agua en suelo)
   - Identifica condiciones desfavorables
   - Reduce falsos positivos en condiciones de sequía

## Próximos Pasos

1. ✅ Implementar cálculo de fecha de inicio de primavera
2. ✅ Implementar cálculo de GDD
3. ✅ Implementar cálculo de días de agua disponible
4. ✅ Integrar con el modelo de predicción
5. ⏭️ Reentrenar el modelo con las nuevas características
6. ⏭️ Evaluar mejora en precisión de predicciones

## Contacto

Para preguntas o problemas, consulta el repositorio principal del proyecto.
