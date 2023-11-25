# MC-RayTracing
Este es un RayTracer hecho en CUDA, una variante de C++ que permite el uso de recursos de tarjetas gráficas. Implementa un Diorama sencillo de Minecraft, simula un entorno de aldea con luces artificiales de antorchas y la luz de la luna.

![](https://github.com/hsilv/MC-RayTracing/blob/main/raytracer.gif)

## Escenografía
Esta escena posee modelos de cubos representados en distintas formas y medidas

### Distribución de bloques
* Los bloques básicos son bloques de medidas 1x1x1.
* Las escaleras son casos especiales, cada escalera posee tres bloques, de medidas 0.5x1x1, 1x0.5x1 o 1x1x0.5. Esto dependerá de la orientación de la escalera.
* Las esquinas de las escaleras, también son casos especiales, poseen dos bloques únicamente, uno mide 1x0.5x1 y el otro mide 0.5x0.5x0.5.
* Los portales son similares a bloques básicos, únicamente que poseen medidas de 0.2x1x1.

### Materiales
* **Pasto:** Es el fondo del diorama, este posee albedo.
* **Madera:** Conforma el techo y el piso de la casa, posee un albedo muy bajo.
* **Troncos:** Conforma las vigas de la casa, al igual que la madera, posee albedo muy bajo.
* **Roca:** Conforma las paredes de la casa y las gradas de entrada, al igual que los materiales anteriores.
* **Portal del Nether:** Este material posee un albedo muy bajo, transparencia y refractividad baja. Sin embargo, posee **reflectividad** de 0.5.
  
  ![](https://github.com/hsilv/MC-RayTracing/blob/main/portal.gif)
* **Cristal:** Conforman las ventanas de la casa. Posee un albedo intermedio entre los materiales anteriores, es reflectivo en 0.1. Y es totalmente transparente y **refractivo** a razón de 0.5.
  
  ![](https://github.com/hsilv/MC-RayTracing/blob/main/glass.gif)

Cabe destacar que todos los materiales anteriores tienen las propiedades reflectivas y refractivas, sin embargo, los materiales sólidos y opacos, como la madera, pasto, troncos y roca. Este efecto no es imperceptible.

### Luces
Esta escena tiene dos luces, una enfrente de la casa, de color anaranjado con una intensidad de 0.5.

![](https://github.com/hsilv/MC-RayTracing/blob/main/torch.png)

Y una luz de luna, puesta detrás de la casa, de color azul, con una intensidad de 0.2.

![](https://github.com/hsilv/MC-RayTracing/blob/main/moon.png)
