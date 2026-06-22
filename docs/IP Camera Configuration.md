# Гайд: IP-камеры + удалённый доступ через Tailscale

## Часть 1 — Подключение камеры к ПК

1. Подключи камеру к ПК через USB-Ethernet адаптер (LAN-кабель)
2. Задай адаптеру статический IP в той же подсети что камера:

   **Linux:**
   ```bash
   sudo ip addr add 192.168.1.50/24 dev eth1
   ```
   **Windows:**
   ```cmd
   netsh interface ipv4 set address name="Ethernet X" static 192.168.1.51 255.255.255.0
   ```
   
3. Зайди в веб-интерфейс камеры (`http://192.168.1.108` — дефолт для Dahua и многих других)
4. В настройках камеры задай:
   - **IP** — уникальный IP-адрес.
   - **Gateway = IP адаптера ПК** (например `192.168.1.50` / `192.168.1.51`) — без этого камера не сможет отвечать на запросы извне своей подсети
5. Проверь локально:
   ```bash
   ping 192.168.1.108
   ```

## Часть 2 — Настройка Tailscale на ПК с камерой

### Linux

```bash
curl -fsSL https://tailscale.com/install.sh | sh

sudo sysctl -w net.ipv4.ip_forward=1
echo 'net.ipv4.ip_forward = 1' | sudo tee -a /etc/sysctl.conf

sudo tailscale up --advertise-routes=192.168.1.108/32 --accept-risk=lose-ssh
```

### Windows

```cmd
:: установка — https://tailscale.com/download

reg add HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters /v IPEnableRouter /t REG_DWORD /d 1 /f
netsh interface ipv4 set interface "Ethernet X" forwarding=enabled
netsh interface ipv4 set interface "Tailscale" forwarding=enabled
```
Возможно потребуется перезагрузка ПК.

```cmd
tailscale up --advertise-routes=192.168.1.108/32 --accept-routes
```

### Добавление каждой следующей камеры

Маршрут для новой камеры (на этом же ПК или на другом) — это всегда отдельный `/32`. Дополняй список через `tailscale set`:

```bash
tailscale set --advertise-routes=192.168.1.108/32,192.168.1.116/32
```

### Одобрить маршрут
`https://login.tailscale.com/admin/machines` → найди устройство → **Edit route settings** → включи галочку на маршруте.

## Часть 3 — Доступ с удалённой машины

1. Установи Tailscale на удалённой машине, авторизуйся
2. Прими маршруты:
   ```bash
   sudo tailscale up --accept-routes
   ```
3. Обращайся к камере напрямую по её локальному IP:
   ```bash
   ping 192.168.1.108
   ffplay rtsp://192.168.1.108:554/<путь_стрима>
   ```

## Чеклист, если что-то не работает

- [ ] На камере задан `Gateway` = IP адаптера ПК
- [ ] IP камеры уникален во всём тайлнете
- [ ] На адаптере ПК нет лишнего default gateway, если в подсети нет настоящего роутера
- [ ] Windows: после правки `IPEnableRouter` был перезапуск
- [ ] Маршрут одобрен в admin console
- [ ] Новая камера добавлена через `tailscale set`, а не через повторный `tailscale up`
- [ ] Если всё ещё не пингуется удалённо — проверь профиль сети адаптера (Public/Private) и правила форвардинга в Windows Defender Firewall